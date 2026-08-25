"""
Discrete-event NTN simulator with queues, key pools, and Age-of-Secret
routing.  Companion to Paper 2 (MILCOM 2026, sole author Liang Dong).

The simulator runs in 1-second steps.  Each step:

  1. Refresh classical link availability from the Walker constellation
     pass-window schedule.
  2. Inject classical capacity / latency for available edges.
  3. Generate keys: QKD on edges with a pass in progress (ungated, the
     entropy is free), plus demand-gated PQC refresh up to a reserve
     sized to the edge's own recent consumption.
  4. Inject traffic arrivals at flow sources.
  5. Compute per-link routing/scheduling decision under the active
     algorithm (AoS-backpressure or one of four baselines).
  6. Drain queues + consume keys for served packets.
  7. Log per-step state for offline analysis.

The six schedulers compared in the paper.  Every one of them is a Dijkstra
over a scheduler-specific edge weight, except aos_ideal, so the comparison
varies only the weight function:

  * shortest_path    - Dijkstra by hop count; key-unaware.
  * pqc_only         - Dijkstra by latency over all edges; key-unaware.
                       This is the deployable PQC-today baseline.  It does
                       not refuse QKD-supplied keys: only qkd_only changes
                       the key supply, by disabling PQC refresh.
  * qkd_only         - Dijkstra by latency over QKD-capable edges only,
                       with PQC refresh disabled.
  * key_rate_aware   - greedy on the largest current key pool K_e.
  * aos_ideal        - the per-edge per-flow max-weight scheduler of the
                       achievability theorem, using the virtual key-queue.
  * aos_cg           - AoS-BP, the paper's MAIN algorithm.  The per-slot
                       max-weight problem over path actions is a linear
                       program with one column per (flow, path); it is
                       solved EXACTLY by column generation, whose pricing
                       subproblem is a Dijkstra run on the drift-derived
                       edge cost psi_e plus the master's edge duals.
                       Fractional splitting across paths within a slot is
                       what makes the LP the exact max-weight problem
                       rather than a relaxation of it.
  * aos_greedy       - ABLATION.  The per-flow rate-aware greedy: process
                       flows in decreasing source backlog, give each its
                       best bottleneck-restricted path, decrement the
                       shared residual capacity and key pools.  This is
                       what earlier drafts of this work analysed.  It is
                       NOT a bounded-loss approximation of the max-weight
                       action: the flows are coupled through the shared
                       key budget, and the loss is unbounded (see
                       `param_study.py` study G and the counterexample in
                       the paper).  Retained to measure the gap.
  * aos_backpressure - AoS-BP-H, the cost-minimising heuristic of the
                       MILCOM version, retained as an ABLATION.  No
                       approximation constant exists for it: without the
                       positivity gate the drift damage grows with Z.

`max_throughput` is also accepted as a legacy alias for latency-Dijkstra;
it is not part of the reported six.

Author: Liang Dong, MILCOM 2026 Paper 2.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import random
import time
from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linprog

from constellation import (
    DEFAULT_GROUND_STATIONS, DEFAULT_SHELL, GroundStation,
    CLEAR_SKY_TAU_ZENITH, build_satellites, cloud_transmittance,
    constellation_epoch_tt_jd,
    load_real_or_synthetic, make_walker_tles, passes, qkd_rate_bps,
)
from skyfield.api import load


# ---------------------------------------------------------------------------
# Network topology
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Edge:
    src: str
    dst: str
    capacity_bps: float        # classical link capacity
    latency_ms: float          # classical end-to-end propagation
    qkd_capable: bool = False  # whether QKD can generate keys on this link
    k_min_bits: float = 50_000        # policy reserve (50 kb default)
    # Key-buffer cap.  With demand-gated PQC refresh this is the smallest
    # value on the sweep grid at which key material generated on
    # traffic-carrying edges is never discarded, at BOTH the 600- and
    # 3600-slot horizons.  Ungated refresh needed 5 Gb for the same
    # property, because idle edges accrue keys indefinitely; gating cuts
    # the requirement by an order of magnitude.  The MILCOM submission
    # used 5 Mb, inside the regime where the cap distorts routing.
    k_max_bits: float = 500_000_000    # 500 Mb (62.5 MB) per edge
    pqc_refresh_bps: float = 200_000  # PQC refresh: ~ML-KEM throughput on 1 core

    def key(self) -> tuple[str, str]:
        return (self.src, self.dst)


# 1 session-key bit covers this many protected payload bits (AES-256
# rekeyed once per 256 packets ~ 32 bytes key per 256*1500 B = 384000 B).
KEY_BITS_PER_PAYLOAD_BIT = 1.0 / 256.0


def default_topology() -> tuple[list[str], list[Edge]]:
    """Tactical-edge NTN topology, classical data plane only.

    Nodes: six ground stations + four regional gateways.  All edges are
    classical and carry packets.  QKD is a *key-supply overlay* attached
    to ground stations: keys generated by Walker-Delta passes refresh
    the GS-incident edges' key pools (see `qkd_capable=True` flag) but
    QKD is not a routing hop.
    """
    gs_names = [g.name for g in DEFAULT_GROUND_STATIONS]
    gw_names = ["GW-NAEast", "GW-NAWest", "GW-EU", "GW-APAC"]
    nodes = gs_names + gw_names

    edges: list[Edge] = []
    # Terrestrial gateway-mesh (high capacity, low latency)
    gw_pairs = list(combinations(gw_names, 2))
    for a, b in gw_pairs:
        edges.append(Edge(a, b, capacity_bps=10e9, latency_ms=45.0))
        edges.append(Edge(b, a, capacity_bps=10e9, latency_ms=45.0))

    # Each GS is multi-homed: one primary gateway (low latency) and a
    # secondary one (higher latency) for routing diversity.  Both edges
    # receive QKD refresh from the GS's overhead passes.
    gs_attach = {
        "Waco-TX":        [("GW-NAWest", 20.0), ("GW-NAEast", 40.0)],
        "FortBragg-NC":   [("GW-NAEast", 18.0), ("GW-NAWest", 42.0)],
        "RamsteinAB-DE":  [("GW-EU",     16.0), ("GW-NAEast", 80.0)],
        "Yokota-JP":      [("GW-APAC",   22.0), ("GW-NAWest", 90.0)],
        "Diego-Garcia":   [("GW-APAC",   30.0), ("GW-EU",     70.0)],
        "CampLemonnier":  [("GW-EU",     26.0), ("GW-APAC",   85.0)],
    }
    for gs, attachments in gs_attach.items():
        for gw, latency in attachments:
            edges.append(Edge(gs, gw, capacity_bps=1e9, latency_ms=latency,
                              qkd_capable=True))
            edges.append(Edge(gw, gs, capacity_bps=1e9, latency_ms=latency,
                              qkd_capable=True))

    return nodes, edges


# ---------------------------------------------------------------------------
# Flow definition
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Flow:
    name: str
    src: str
    dst: str
    arrival_bps: float                # mean classical-payload arrival
    security_class: str = "session"   # session / command / bulk
    aos_max: float = 60.0             # seconds: freshness target
    key_bits_per_packet: float = 256  # AES-256 session-key cost per packet


def default_flows() -> list[Flow]:
    return [
        Flow("cmd-A", "Waco-TX",       "RamsteinAB-DE", 30e6, "command", aos_max=20),
        Flow("cmd-B", "FortBragg-NC",  "Yokota-JP",     25e6, "command", aos_max=20),
        Flow("ses-A", "RamsteinAB-DE", "Diego-Garcia",  50e6, "session", aos_max=60),
        Flow("ses-B", "Yokota-JP",     "Waco-TX",       40e6, "session", aos_max=60),
        Flow("bulk-A","CampLemonnier", "FortBragg-NC", 100e6, "bulk",    aos_max=120),
    ]


# ---------------------------------------------------------------------------
# Pre-computed pass-window schedule
# ---------------------------------------------------------------------------

_ERA5_CACHE: dict | None = None


def load_era5_tau(path: str | None = None) -> dict | None:
    """Hourly slant-column optical depth per ground station, from ERA5.

    Returns a dict station -> numpy array of tau on a regular hourly
    grid, or None if the derived file has not been built.  Produced by
    `fetch_era5.py` followed by `era5_to_transmittance.py`.
    """
    global _ERA5_CACHE
    if _ERA5_CACHE is not None:
        return _ERA5_CACHE
    import pandas as pd
    p = Path(path) if path else Path(__file__).resolve().parents[1] \
        / "data" / "era5_transmittance.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["time"])
    _ERA5_CACHE = {s: (g.sort_values("time")["tau"].to_numpy(),
                       g.sort_values("time")["tcc"].to_numpy())
                   for s, g in df.groupby("station")}
    return _ERA5_CACHE


def era5_transmittance(era5: dict, station: str, hours_from_start: float,
                       elev_deg: float, week_offset: int = 0,
                       rng: np.random.Generator | None = None) -> float:
    """Beer-Lambert transmittance on the reanalysis column at a pass.

    A quarter-degree cell is far wider than a QKD beam, so the grid-box
    MEAN optical depth must not be applied to the beam as though the
    cloud were uniform.  Doing so is a Jensen error and it is severe:
    transmittance is convex in optical depth, and at these sites the
    mean depth is dominated by the thick tail, so a uniform column makes
    a partly cloudy hour look opaque when a beam through the clear
    fraction would get through untouched.

    The beam is therefore treated as intersecting cloud with probability
    tcc, the reanalysis cloud fraction, and passing through clear air
    otherwise.  In-cloud optical depth is recovered from the grid-box
    mean by dividing out the fraction.  Temporal correlation survives
    because tcc is itself strongly autocorrelated.

    The reanalysis is hourly and a pass lasts a few minutes, so the
    containing hour is used without interpolation.
    """
    rec = era5.get(station)
    if rec is None or len(rec[0]) == 0 or elev_deg <= 0.0:
        return 0.0
    tau_series, tcc_series = rec
    i = (int(hours_from_start) + week_offset * 168) % len(tau_series)
    tau_tot = float(tau_series[i])
    frac = min(max(float(tcc_series[i]), 0.0), 1.0)
    tau_cloud = max(tau_tot - CLEAR_SKY_TAU_ZENITH, 0.0)
    airmass = 1.0 / math.sin(math.radians(elev_deg))
    rng = rng or np.random.default_rng(i)
    if frac > 1e-3 and rng.random() < frac:
        tau = CLEAR_SKY_TAU_ZENITH + tau_cloud / frac     # in-cloud depth
    else:
        tau = CLEAR_SKY_TAU_ZENITH
    return float(math.exp(-tau * airmass))


def build_qkd_schedule(start_jd: float | None = None, hours: float = 12.0,
                       min_elev_deg: float = 25.0,
                       max_sats: int | None = None,
                       weather_seed: int = 0,
                       prefer_real: bool = True,
                       weather_model: str = "isccp"
                       ) -> tuple[dict, str]:
    """Return ``(schedule, provenance)``.

    ``schedule`` is a per-ground-station list of
    ``(t_start, t_end, peak_elev, average-rate-bps)`` tuples spanning the
    requested window.  ``provenance`` is a short string describing whether
    the underlying constellation came from a real TLE snapshot or the
    synthetic Walker-Delta fallback, and at which epoch it was propagated.

    ``start_jd`` is a TT Julian date.  When it is ``None`` (the default) the
    window is anchored at the median TLE epoch of the loaded snapshot, which
    is the only defensible choice: SGP4 error grows rapidly with time from
    epoch, so propagating to an unrelated fixed date does not reproduce the
    snapshot's actual orbital state.

    Per-pass optical attenuation comes from the Beer-Lambert slant-path
    model in ``constellation.cloud_transmittance`` (ISCCP mid-latitude cloud
    climatology), drawn once per pass from ``weather_seed``.
    """
    rng = np.random.default_rng(weather_seed)
    tles, provenance = load_real_or_synthetic(prefer_real=prefer_real)
    if max_sats is not None:
        tles = tles[:max_sats]
    sats = build_satellites(tles)
    ts = load.timescale()
    if start_jd is None:
        start_jd = constellation_epoch_tt_jd(sats)
    t_start = ts.tt_jd(start_jd)
    t_end   = ts.tt_jd(start_jd + hours / 24.0)
    provenance += (f"  Propagated from TT JD {start_jd:.5f} "
                   f"({t_start.utc_strftime('%Y-%m-%d %H:%M UTC')}), "
                   f"{hours:g} h window, min elevation {min_elev_deg:g} deg.")
    era5 = load_era5_tau() if weather_model == "era5" else None
    # Spread the seeds evenly over the reanalysis year rather than
    # over consecutive weeks: correlated cloud has an e-folding time
    # of 9 to 14 h here, so adjacent weeks are nearly independent but
    # consecutive ones still share a season.
    week_offset = (weather_seed * 4) % 52
    if era5 is not None:
        # Each seed reads a different week of the reanalysis year, so
        # seeds vary the weather realization while the orbital geometry
        # stays anchored at the element-set epoch.
        provenance += (f"  Cloud from ERA5 reanalysis, week offset "
                       f"{week_offset}.")
    sched = {g.name: [] for g in DEFAULT_GROUND_STATIONS}
    for sat in sats:
        for gs in DEFAULT_GROUND_STATIONS:
            for rise_tt, set_tt, peak_elev in passes(sat, gs, t_start, t_end,
                                                     min_elev_deg=min_elev_deg):
                if era5 is not None:
                    # Beer-Lambert on the reanalysis slant column, so
                    # consecutive passes over a site inherit the real
                    # persistence of the cloud field instead of being
                    # redrawn independently.
                    hours = (0.5 * (rise_tt + set_tt) - start_jd) * 24.0
                    weather = era5_transmittance(
                        era5, gs.name, hours, peak_elev,
                        week_offset=week_offset, rng=rng)
                else:
                    weather = cloud_transmittance(peak_elev, rng=rng)
                avg_rate = qkd_rate_bps(peak_elev, weather=weather) * 0.6
                t0_s = (rise_tt - start_jd) * 86400.0
                t1_s = (set_tt  - start_jd) * 86400.0
                if t1_s > t0_s:
                    sched[gs.name].append((t0_s, t1_s, peak_elev, avg_rate))
    for gs in sched:
        sched[gs].sort()
    return sched, provenance


def qkd_rate_at(t_s: float, schedule_for_gs: list[tuple]) -> float:
    """Return aggregate QKD generation rate (bps) at simulation time t_s."""
    r = 0.0
    for t0, t1, _, rate in schedule_for_gs:
        if t0 <= t_s <= t1:
            r += rate
    return r


# ---------------------------------------------------------------------------
# Network simulator
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SimConfig:
    horizon_s: int = 600
    dt_s: int = 1
    seed: int = 0
    pqc_handshake_bits: float = 8192    # cost of one ML-KEM handshake
    aos_lambda: float = 2.0             # AoS depletion-risk weight
    aos_mu: float = 0.5                 # AoS trust-risk weight
    bp_alpha: float = 5.0e-7            # backpressure key-gradient weight
    bp_beta: float = 1.0e-3             # backpressure latency penalty
    bp_chi: float = 5.0e-3              # backpressure AoS penalty
    bp_omega: float = 2.0e2             # Lyapunov Z weight in ideal max-weight
    scheduler: str = "aos_backpressure"
    scenario: str = "nominal"
    load_scale: float = 1.0             # multiplies every flow's arrival rate,
                                        # for sweeping the load across Lambda_S
    # Eq. (2) caps.  Both are required for AoS_e to be bounded, which is the
    # bounded-penalty hypothesis Theorem 1 assumption (i) needs; earlier
    # revisions left the age term uncapped, so the hypothesis did not hold
    # in the simulator it was validated against.
    aos_T_a: float = 300.0              # cap on the age term t - g_e, seconds
    aos_tau_d: float = 60.0             # cap on the dimensionless depletion ratio
    k_max_bits: float | None = None     # override every Edge.k_max_bits when set
    # Warm-start pool level, in bits, independent of k_max.  2.5 Mb is the
    # level the MILCOM submission started from (12.5 s of PQC refresh).
    k_init_bits: float = 2_500_000
    # Demand-gated PQC refresh.  Running key establishment into a pool that
    # is already adequate burns CPU for material that is never spent, and
    # it makes the overflow audit horizon-dependent: an idle edge saturates
    # ANY finite k_max given a long enough run.  With gating, an edge
    # maintains a reserve sized to its own recent demand and stops there,
    # so k_max only has to absorb QKD pass bursts above that reserve.
    # QKD is deliberately NOT gated: the pass is happening regardless and
    # the entropy is free, so the asymmetry is physical, not a modelling
    # convenience.
    pqc_gated: bool = True
    pqc_reserve_s: float = 60.0     # target reserve, in seconds of demand
    pqc_ewma_slots: float = 30.0    # demand estimator time constant
    # Form of the key-scarcity term in the Algorithm 1 Dijkstra weight,
    # Eq. (8).  W3 is the selected form: it is the only candidate whose
    # mean AoS is invariant to K_max across four decades at both the 600 s
    # and 3600 s horizons.  See `_key_term`.
    weight_variant: str = "W3"
    # Penalty weights for Algorithm 1' (`aos_bp_prime`).  These are on a
    # different scale from bp_beta/bp_chi by construction: Algorithm 1
    # minimises an additive path cost, in which the weights compete only
    # with each other, whereas Algorithm 1' maximises the rate-weighted
    # objective R(P)*(Q_src - sum psi_e), in which they compete against the
    # source backlog Q_src (order 1e8 bits here).  Selected by sweep:
    # chi' at the knee of the utility-delay curve (minimum mean AoS subject
    # to zero backlog growth and full goodput), beta' at the largest decade
    # with no degradation, two decades below the cliff at 1e5.
    bp_chi_prime: float = 5.0e3
    bp_beta_prime: float = 1.0e3
    # Column generation for the exact per-slot max-weight LP (`aos_cg`).
    # Termination is certified by the pricing step, not by the iteration
    # cap: the loop stops when no (flow, path) column prices out, which is
    # LP optimality.  The cap is a safety net and `cg_converged` records
    # whether it was ever reached.
    cg_max_iters: int = 30
    cg_tol: float = 1e-7            # reduced-cost tolerance, in Mb-units
    cg_pool_per_flow: int = 24      # persistent warm-start columns per flow


# Scale used inside the master LP.  Rates and weights are both carried in
# units of 1e6 bits, which keeps the constraint matrix, the cost vector,
# and the duals within three decades of unity.
LP_SCALE = 1.0e6


def neighbours(edges: list[Edge]) -> dict[str, list[Edge]]:
    out = defaultdict(list)
    for e in edges:
        out[e.src].append(e)
    return out


def shortest_paths(nodes: list[str], edges: list[Edge],
                   metric: str = "hops") -> dict:
    """Single-source shortest paths from every node, by chosen metric.

    Ties are broken on the node name, not on set iteration order.  Python
    randomizes string hashing per process, so `min` over a set of node
    names would pick a different tied node from run to run; with
    hop-count weights every path of equal length ties, so the route was
    genuinely process-dependent.  The chosen routes were always among
    exactly interchangeable edges and no reported metric moved, but the
    per-run logs were not bit-reproducible, which for a released
    artifact is worth eliminating.
    """
    nbr = neighbours(edges)
    sps = {}
    for s in nodes:
        dist = {n: math.inf for n in nodes}
        prev = {n: None for n in nodes}
        dist[s] = 0
        unvisited = set(nodes)
        while unvisited:
            u = min(unvisited, key=lambda x: (dist[x], x))
            if dist[u] == math.inf:
                break
            unvisited.remove(u)
            for e in nbr.get(u, []):
                w = 1 if metric == "hops" else e.latency_ms
                d = dist[u] + w
                if d < dist[e.dst]:
                    dist[e.dst] = d
                    prev[e.dst] = u
        sps[s] = prev
    return sps


def path(prev: dict, src: str, dst: str) -> list[str]:
    out = []
    cur = dst
    while cur is not None:
        out.append(cur)
        if cur == src:
            break
        cur = prev[cur]
    return list(reversed(out)) if out and out[-1] == src else []


@dataclasses.dataclass
class StepLog:
    t: int
    total_secure_bits: float
    total_violations: int
    keypool_min_bits: float
    keypool_mean_bits: float
    queue_total_bits: int          # L1 norm of Q, in bits (CSV column q_total)
    aos_mean: float
    aos_max: float
    rekey_events: int
    # Joint-Lyapunov instrumentation (Eq. (6)).  q_sumsq and z_sumsq are the
    # *sums of squares* that L(X) is actually built from; the L1 norms above
    # and in z_l1 are what the strong-stability conclusion bounds.
    q_sumsq: float = 0.0
    z_l1: float = 0.0
    z_sumsq: float = 0.0
    z_max: float = 0.0
    lyapunov: float = 0.0
    # Nominal key inflow this cycle (QKD + PQC, before the K_max cap), and
    # the part of it discarded at the cap.  Theorem 1 assumption (iii)
    # asserts the discarded fraction is negligible; logging both makes that
    # auditable instead of assumed.
    key_inflow_bits: float = 0.0
    key_overflow_bits: float = 0.0
    # Column-generation instrumentation.  `obj_lp` is the exact optimum of
    # the per-slot max-weight LP; `obj_greedy` is what the per-flow greedy
    # would have achieved from the same state.  Their ratio is the
    # empirical counterpart of the unbounded worst-case gap.
    cg_iters: int = 0
    cg_columns: int = 0
    cg_converged: int = 1
    obj_lp: float = 0.0
    obj_greedy: float = 0.0
    # Slots in which the physical pool, not the virtual queue, was the
    # binding key constraint on some edge carrying traffic.
    key_blocked_edges: int = 0


class AoSNetwork:
    def __init__(self, nodes, edges, flows, cfg: SimConfig, qkd_schedule):
        self.nodes = nodes
        self.edges = edges
        self.flows = flows
        self.cfg = cfg
        self.qkd_schedule = qkd_schedule

        # Per-flow per-node queue (FIFO of packet sizes in bits)
        self.Q: dict[tuple[str, str], deque] = {(f.name, n): deque()
                                                 for f in flows for n in nodes}
        # Per-edge key pool (bits).  The warm-start level is an absolute
        # quantity, NOT a fraction of k_max.  Scaling it with the cap (the
        # earlier 0.5*(k_min+k_max)) makes the initial endowment grow with
        # the buffer, so a large cap hands the network more free key
        # material than an entire run consumes and switches off key scarcity
        # altogether.  That confounds any sensitivity study of k_max.
        self.K: dict[tuple[str, str], float] = {
            e.key(): min(e.k_max_bits, cfg.k_init_bits) for e in edges
        }
        # Per-edge generation epoch (sim seconds since last refresh)
        self.gen_epoch: dict[tuple[str, str], float] = {
            e.key(): 0.0 for e in edges
        }
        # Per-edge virtual key-queue Z_ij of Eq. (5):
        #     Z(t+1) = [Z(t) + rho*D(t) - G(t)]^+
        # where D is the data bits served on the edge this cycle and G is the
        # per-cycle key inflow.  Z accumulates exactly when key consumption
        # outruns key supply, so its mean-rate stability is equivalent to
        # satisfying the average key-refresh constraint of Lambda_S.  This is
        # a *virtual* queue: it is a measurement of constraint violation, not
        # a physical buffer, and it is tracked for every scheduler so that
        # the joint Lyapunov function is comparable across them.
        self.Z: dict[tuple[str, str], float] = {e.key(): 0.0 for e in edges}
        # Per-cycle accumulators, reset at the top of every step().
        self._G_cycle: dict[tuple[str, str], float] = defaultdict(float)
        self._D_cycle: dict[tuple[str, str], float] = defaultdict(float)
        self._overflow_cycle: dict[tuple[str, str], float] = defaultdict(float)
        # Cumulative per-edge totals, so overflow can be reported restricted
        # to edges that actually carried traffic.  Assumption (iii) is about
        # buffering supply burstiness on edges in use: a permanently idle
        # edge saturates at any finite K_max given a long enough horizon, so
        # a whole-network overflow target is not a well-posed test of it.
        self.cum_G: dict[tuple[str, str], float] = defaultdict(float)
        self.cum_D: dict[tuple[str, str], float] = defaultdict(float)
        self.cum_overflow: dict[tuple[str, str], float] = defaultdict(float)
        self.cum_generated: dict[tuple[str, str], float] = defaultdict(float)
        # EWMA of per-slot key consumption, used as the demand signal for
        # the PQC gate.  Seeded at the reserve floor so a cold edge is not
        # starved on its first burst.
        self.u_hat: dict[tuple[str, str], float] = {
            e.key(): e.k_min_bits / max(1.0, cfg.pqc_reserve_s)
            for e in edges
        }
        self._gen_cycle: dict[tuple[str, str], float] = defaultdict(float)
        self.rng = np.random.default_rng(cfg.seed)
        self.rekey_events = 0

        # Precompute shortest-paths (for static baselines)
        self.sp_hops    = shortest_paths(nodes, edges, "hops")
        self.sp_latency = shortest_paths(nodes, edges, "latency")

        # Index edges by endpoint
        self.edge_index = {e.key(): e for e in edges}
        self.out_edges  = neighbours(edges)

        # Logging
        self.logs: list[StepLog] = []

        # Warm-start column pool for `aos_cg`, one list of paths per flow.
        # Reusing columns across slots only changes how many pricing rounds
        # the loop needs; optimality is certified by the pricing step, so
        # the returned action is the exact LP optimum either way.
        self.col_pool: dict[str, list[tuple]] = {f.name: [] for f in flows}
        self.cg_nonconverged = 0

    # -----------------------------------------------------------------
    def edge_aos(self, e: Edge, t: int) -> float:
        """Eq. (2): AoS_e = min(T_a, t - g_e) + lambda_a * min(tau_d, u/(K+eps)).

        Both caps are load-bearing.  They give the bound
        AoS_e <= T_a + lambda_a * tau_d (+ mu_a with a nonzero trust score),
        which is exactly the finite-penalty condition Theorem 1 assumption
        (i) requires, and they make the metric normalisable.
        """
        cfg = self.cfg
        k = self.K[e.key()]
        u_forecast = 1e6           # 1 Mb/s forecast for the next cycle
        age = min(cfg.aos_T_a, t - self.gen_epoch[e.key()])
        depletion = min(cfg.aos_tau_d, u_forecast / (k + 1e3))
        return age + cfg.aos_lambda * depletion

    def aos_bound(self) -> float:
        """The A_ref = T_a + lambda_a*tau_d bound that `edge_aos` cannot
        exceed.  Used to normalise the AoS term of the routing weight."""
        return self.cfg.aos_T_a + self.cfg.aos_lambda * self.cfg.aos_tau_d

    # -----------------------------------------------------------------
    def _key_term(self, e: Edge, k: float) -> float:
        """Key-scarcity term of the Algorithm 1 Dijkstra weight, Eq. (8).

        Four candidate forms.  W0 is what the MILCOM submission used; it is
        in absolute key-bits, so its magnitude rescales with K_max while the
        latency and AoS terms do not.  That makes the routing metric depend
        on the buffer size, and at large K_max it swamps the AoS term and
        collapses Algorithm 1 into the Key-rate-aware baseline.

          W0  alpha * (K_max - K)                      absolute key-bits
          W1  alpha1 * (1 - K/K_max)                   normalised to buffer
          W2  (none)                                   AoS carries scarcity
          W3  alpha3 * min(tau_d, rho*C*dt / (K+eps))  normalised to demand

        Calibration rule, fixed before any variant was run: each term is
        scaled to equal W0 at K = K_max/2, the simulator's initial pool
        level, using the reference configuration (K_max = 5 Mb, a 1 Gbps
        QKD-capable edge).  W0 there is 5e-7 * 2.5e6 = 1.25.
        """
        cfg = self.cfg
        v = cfg.weight_variant
        if v == "W0":
            return cfg.bp_alpha * (e.k_max_bits - k)
        if v == "W1":
            return 2.5 * (1.0 - k / e.k_max_bits)
        if v == "W2":
            return 0.0
        if v == "W3":
            rho = KEY_BITS_PER_PAYLOAD_BIT
            return 0.8 * min(cfg.aos_tau_d,
                             rho * e.capacity_bps * cfg.dt_s / (k + 1e3))
        raise ValueError(f"unknown weight_variant {v!r}")

    def step(self, t: int) -> StepLog:
        cfg = self.cfg

        # ------ scenario perturbations ------
        scenario = cfg.scenario
        weather_mult = 1.0
        relay_block = set()
        if scenario == "weather":
            weather_mult = 0.2 if (t // 120) % 2 == 0 else 1.0
        elif scenario == "relay_compromise":
            if 200 <= t < 400:
                relay_block.add("GW-EU")
        elif scenario == "traffic_surge":
            pass   # handled below
        elif scenario == "coalition_partition":
            if 250 <= t < 500:
                relay_block.add("GW-APAC")

        # Reset the per-cycle key-flow accounting that feeds Eq. (5).
        self._G_cycle.clear()
        self._D_cycle.clear()
        self._overflow_cycle.clear()
        self._gen_cycle.clear()

        def deposit(ekey: tuple[str, str], e: Edge, capability: float,
                    gated: bool = False) -> None:
            """Offer `capability` key-bits to edge `e`.

            G_ij(t) in Eq. (5), and hence Gbar_ij in Lambda_S, is the key
            *capability*: what the edge could have produced this slot.  It
            is recorded whether or not the gate suppresses generation.
            That choice is what keeps Lambda_S policy-independent -- if the
            region were defined on gated inflow it would depend on the
            scheduler being analysed, and the converse would be vacuous.

            The physical pool receives only what is actually generated.
            """
            self._G_cycle[ekey] += capability
            if gated and self.cfg.pqc_gated:
                target = min(e.k_max_bits,
                             max(e.k_min_bits,
                                 self.cfg.pqc_reserve_s * self.u_hat[ekey]))
                amount = min(capability, max(0.0, target - self.K[ekey]))
            else:
                amount = capability
            if amount <= 0:
                return
            before = self.K[ekey]
            self.K[ekey] = min(e.k_max_bits, before + amount)
            self._gen_cycle[ekey] += amount
            self._overflow_cycle[ekey] += max(0.0, amount
                                              - (self.K[ekey] - before))

        # ------ 1. QKD key generation (per-GS rate refreshes GS-incident edges)
        for gs in [g.name for g in DEFAULT_GROUND_STATIONS]:
            rate = qkd_rate_at(t, self.qkd_schedule.get(gs, [])) * weather_mult
            if rate <= 0:
                continue
            for e in self.edges:
                if e.qkd_capable and (e.src == gs or e.dst == gs):
                    deposit(e.key(), e, rate * cfg.dt_s)
                    self.gen_epoch[e.key()] = float(t)

        # ------ 2. PQC background refresh ------
        for e in self.edges:
            if e.pqc_refresh_bps > 0:
                if cfg.scheduler == "qkd_only":
                    continue   # QKD-only refuses PQC refresh
                deposit(e.key(), e, e.pqc_refresh_bps * cfg.dt_s, gated=True)

        # ------ 3. Traffic arrivals ------
        violation_count = 0
        for f in self.flows:
            arrival = f.arrival_bps * cfg.dt_s * cfg.load_scale
            if scenario == "traffic_surge" and 200 <= t < 400:
                arrival *= 3.0
            self.Q[(f.name, f.src)].append(arrival)

        # ------ 4. Scheduling -------------------------------------------------
        if cfg.scheduler in ("aos_cg", "aos_bp_prime", "aos_greedy"):
            # aos_cg is AoS-BP, the exact per-slot max-weight action.  The
            # other two names select the greedy ablation.
            if cfg.scheduler == "aos_cg":
                r = self._step_aos_cg(t, relay_block)
            else:
                r = self._step_alg1_prime(t, relay_block)
            violation_count += r["violations"]
            self._advance_virtual_queues()
            log = self._make_log(t, r["bits_delivered"], violation_count,
                                 r["aos_observed"], r["rekey_events"],
                                 cg=r)
            self.logs.append(log)
            return log

        if cfg.scheduler == "aos_ideal":
            # Per-edge per-flow max-weight backpressure: the ideal scheduler
            # of Theorem 1.  Implemented separately because it does not use
            # the per-flow shortest-path abstraction.
            r = self._step_ideal_bp(t, relay_block)
            bits_served    = r["bits_delivered"]
            rekey_events   = r["rekey_events"]
            aos_observed   = r["aos_observed"]
            violation_count += r["violations"]
            self._advance_virtual_queues()
            log = self._make_log(t, bits_served, violation_count,
                                 aos_observed, rekey_events)
            self.logs.append(log)
            return log

        # ------ 4 (legacy). Dijkstra-based per-flow routing ------------------
        bits_served = 0.0
        rekey_events = 0
        aos_observed: list[float] = []
        for f in self.flows:
            # Find a route per scheduler choice
            route = self._route(f, t, relay_block)
            if not route or len(route) < 2:
                continue
            # Determine pipe capacity along the route (min link cap + min key)
            pipe_cap = float("inf")
            for u, v in zip(route, route[1:]):
                e = self.edge_index.get((u, v))
                if e is None:
                    pipe_cap = 0.0
                    break
                pipe_cap = min(pipe_cap, e.capacity_bps * cfg.dt_s)
            # Key budget along the route, scaled to payload-bits served
            key_budget_bits = float("inf")
            for u, v in zip(route, route[1:]):
                e = self.edge_index.get((u, v))
                if e is None:
                    key_budget_bits = 0.0
                    break
                key_budget_bits = min(key_budget_bits, self.K[e.key()])
            key_budget = key_budget_bits / KEY_BITS_PER_PAYLOAD_BIT
            # Pop from source queue, drain through pipe
            avail = sum(self.Q[(f.name, f.src)])
            served = min(avail, pipe_cap, key_budget)
            if served < avail and avail > 0:
                # protected-bit demand exceeds key reserve -> violation
                if key_budget < pipe_cap:
                    violation_count += 1
            # Pop "served" bits from source queue (oldest first)
            remaining = served
            while remaining > 0 and self.Q[(f.name, f.src)]:
                head = self.Q[(f.name, f.src)][0]
                if head <= remaining:
                    remaining -= head
                    self.Q[(f.name, f.src)].popleft()
                else:
                    self.Q[(f.name, f.src)][0] -= remaining
                    remaining = 0
            bits_served += served
            # Consume keys along the route, scaled by KEY_BITS_PER_PAYLOAD_BIT
            key_bits_used = served * KEY_BITS_PER_PAYLOAD_BIT
            for u, v in zip(route, route[1:]):
                e = self.edge_index.get((u, v))
                if e is None: continue
                self.K[e.key()] = max(0.0, self.K[e.key()] - key_bits_used)
                self._D_cycle[e.key()] += served   # D_ij(t) for Eq. (5)
                # Trigger a rekey event if we crossed the policy minimum
                if self.K[e.key()] < e.k_min_bits:
                    rekey_events += 1
                aos_observed.append(self.edge_aos(e, t))

        # ------ 5. Virtual key-queue update and logging ------
        self._advance_virtual_queues()
        log = self._make_log(t, bits_served, violation_count,
                             aos_observed, rekey_events)
        self.logs.append(log)
        return log

    # -----------------------------------------------------------------
    def _advance_virtual_queues(self) -> None:
        """Apply Eq. (5) to every edge once this cycle's service is settled.

            Z_ij(t+1) = [ Z_ij(t) + rho * D_ij(t) - G_ij(t) ]^+

        Called after scheduling so that D_ij(t) is the data actually served
        this cycle, and before logging so the reported Lyapunov value is
        L(X(t+1)) consistently across schedulers.
        """
        rho = KEY_BITS_PER_PAYLOAD_BIT
        for ekey in self.Z:
            drift = rho * self._D_cycle[ekey] - self._G_cycle[ekey]
            self.Z[ekey] = max(0.0, self.Z[ekey] + drift)
            self.cum_G[ekey] += self._G_cycle[ekey]
            self.cum_D[ekey] += self._D_cycle[ekey]
            self.cum_overflow[ekey] += self._overflow_cycle[ekey]
            self.cum_generated[ekey] += self._gen_cycle[ekey]
            # Demand estimator for the PQC gate.
            a = 1.0 / max(1.0, self.cfg.pqc_ewma_slots)
            self.u_hat[ekey] = ((1.0 - a) * self.u_hat[ekey]
                                + a * rho * self._D_cycle[ekey])

    def _make_log(self, t, bits_served, violation_count, aos_observed,
                  rekey_events, cg: dict | None = None):
        keypool_vals = list(self.K.values())
        keypool_min = min(keypool_vals)
        keypool_mean = float(np.mean(keypool_vals))
        queue_bits = [float(sum(q)) for q in self.Q.values()]
        queue_total = sum(int(round(b)) for b in queue_bits)
        q_sumsq = float(sum(b * b for b in queue_bits))
        z_vals = list(self.Z.values())
        z_sumsq = float(sum(z * z for z in z_vals))
        # L(X) of Eq. (6): (1/2)sum_{i,f} (Q^f_i)^2 + (omega/2)sum_{ij} Z_ij^2.
        lyap = 0.5 * q_sumsq + 0.5 * self.cfg.bp_omega * z_sumsq
        return StepLog(
            t=int(t),
            total_secure_bits=float(bits_served),
            total_violations=int(violation_count),
            keypool_min_bits=float(keypool_min),
            keypool_mean_bits=float(keypool_mean),
            queue_total_bits=int(queue_total),
            aos_mean=float(np.mean(aos_observed) if aos_observed else 0.0),
            aos_max=float(max(aos_observed) if aos_observed else 0.0),
            rekey_events=int(rekey_events),
            q_sumsq=q_sumsq,
            z_l1=float(sum(z_vals)),
            z_sumsq=z_sumsq,
            z_max=float(max(z_vals) if z_vals else 0.0),
            lyapunov=lyap,
            key_inflow_bits=float(sum(self._G_cycle.values())),
            key_overflow_bits=float(sum(self._overflow_cycle.values())),
            cg_iters=int((cg or {}).get("iters", 0)),
            cg_columns=int((cg or {}).get("columns", 0)),
            cg_converged=int((cg or {}).get("converged", 1)),
            obj_lp=float((cg or {}).get("obj_lp", 0.0)),
            obj_greedy=float((cg or {}).get("obj_greedy", 0.0)),
            key_blocked_edges=int((cg or {}).get("key_blocked_edges", 0)),
        )

    # -----------------------------------------------------------------
    def _step_ideal_bp(self, t: int, relay_block: set[str]) -> dict:
        """Per-edge per-flow max-weight backpressure (the ideal scheduler
        of Theorem 1).  Each cycle, every admissible edge picks the flow
        with the largest AoS-aware weight and serves it.  Packets that
        do not yet reach their destination accumulate at intermediate
        nodes; on subsequent cycles they continue forward by the same
        max-weight rule.
        """
        cfg = self.cfg
        bits_delivered = 0.0
        rekey_events = 0
        aos_observed: list[float] = []
        violations = 0

        # ----- Phase A. Per-edge max-weight decisions -----
        # omega weights the virtual key-queue Z in the Lyapunov function.
        # The theorem holds for any omega > 0; the chosen value sets the
        # relative emphasis on key-pool stability versus queue stability.
        # We scale it so that omega*rho*K_min ~ typical Q gradient
        # (millions of bits) and the deficit term is binding.
        rho = KEY_BITS_PER_PAYLOAD_BIT
        omega = cfg.bp_omega
        decisions: dict[tuple[str, str], tuple[str, float]] = {}
        for e in self.edges:
            if e.src in relay_block or e.dst in relay_block:
                continue
            k = self.K[e.key()]
            # Virtual key-queue Z_ij of Eq. (5), carried across cycles.  It
            # is the running excess of key consumption over key supply on
            # this edge, so omega*rho*Z is the key-deficit gradient that
            # diverts traffic away from edges outrunning their refresh rate.
            # (An earlier version used the memoryless proxy
            # max(0, K_min - K), which is bounded by K_min by construction
            # and therefore cannot express a sustained deficit at all.)
            z = self.Z[e.key()]
            aos_e = self.edge_aos(e, t)
            best_flow: str | None = None
            best_w = 0.0
            for f in self.flows:
                q_i = sum(self.Q[(f.name, e.src)])
                # Treat destination's queue as a 0-pressure sink.
                if e.dst == f.dst:
                    q_j = 0.0
                else:
                    q_j = sum(self.Q[(f.name, e.dst)])
                queue_grad = q_i - q_j
                # Lagrangian penalty matching Eq. (4) in the paper.
                penalty = (omega * rho * z
                           + cfg.bp_beta  * e.latency_ms
                           + cfg.bp_chi   * aos_e)
                w = queue_grad - penalty
                if w > best_w:
                    best_w = w
                    best_flow = f.name
            if best_flow is not None:
                decisions[e.key()] = (best_flow, best_w)

        # ----- Phase B. Serve activated edges -----
        for ekey, (fname, _w) in decisions.items():
            e = self.edge_index[ekey]
            f = next(fl for fl in self.flows if fl.name == fname)
            avail = sum(self.Q[(fname, e.src)])
            if avail <= 0:
                continue
            pipe = e.capacity_bps * cfg.dt_s
            key_budget = self.K[ekey] / rho
            served = min(avail, pipe, key_budget)
            if served <= 0:
                continue
            if served < min(avail, pipe) and key_budget < pipe:
                violations += 1
            # Drain source queue
            remaining = served
            while remaining > 0 and self.Q[(fname, e.src)]:
                head = self.Q[(fname, e.src)][0]
                if head <= remaining:
                    remaining -= head
                    self.Q[(fname, e.src)].popleft()
                else:
                    self.Q[(fname, e.src)][0] -= remaining
                    remaining = 0
            # Deliver or accumulate at next-hop queue
            if e.dst == f.dst:
                bits_delivered += served
            else:
                self.Q[(fname, e.dst)].append(served)
            # Consume keys
            key_bits_used = served * rho
            self.K[ekey] = max(0.0, self.K[ekey] - key_bits_used)
            self._D_cycle[ekey] += served      # D_ij(t) for Eq. (5)
            if self.K[ekey] < e.k_min_bits:
                rekey_events += 1
            aos_observed.append(self.edge_aos(e, t))

        return dict(bits_delivered=bits_delivered,
                    rekey_events=rekey_events,
                    aos_observed=aos_observed,
                    violations=violations)

    # -----------------------------------------------------------------
    def _dijkstra_psi(self, f: Flow, allowed: set, psi: dict) -> list[str]:
        """Min-cost path for flow f over `allowed` edges with costs `psi`.

        All costs are non-negative (omega*rho*Z >= 0, beta*L > 0,
        chi*AoS >= 0), so Dijkstra is valid without reweighting.
        """
        dist = {n: math.inf for n in self.nodes}
        prev = {n: None for n in self.nodes}
        dist[f.src] = 0.0
        unvisited = set(self.nodes)
        while unvisited:
            u = min(unvisited, key=lambda x: (dist[x], x))
            if dist[u] == math.inf or u == f.dst:
                break
            unvisited.remove(u)
            for e in self.out_edges.get(u, []):
                ek = e.key()
                if ek not in allowed or e.dst not in unvisited:
                    continue
                d = dist[u] + psi[ek]
                if d < dist[e.dst]:
                    dist[e.dst] = d
                    prev[e.dst] = u
        return path(prev, f.src, f.dst)

    # -----------------------------------------------------------------
    # Shared per-slot machinery for the source-routed schedulers.
    # -----------------------------------------------------------------
    def _slot_state(self, t: int, relay_block: set[str]) -> tuple[dict, dict,
                                                                  dict]:
        """Freeze the per-slot data the max-weight problem is posed on.

        Returns ``(rbar, psi, qsrc)`` where ``rbar[e]`` is the per-edge rate
        ceiling min{C_e dt, K_e/rho} in payload bits, ``psi[e]`` is the
        drift-derived edge cost omega*rho*Z_e + beta*L_e + chi*AoS_e, and
        ``qsrc[f]`` is the source backlog.

        psi is evaluated once, at the state X(t), and held fixed for the
        whole slot.  The depletion term of AoS_e depends on K_e, so
        recomputing psi after each flow is served would make the objective
        depend on the order in which flows are considered and would no
        longer be the cost appearing in the drift inequality.
        """
        cfg = self.cfg
        rho = KEY_BITS_PER_PAYLOAD_BIT
        omega = cfg.bp_omega
        rbar: dict[tuple[str, str], float] = {}
        psi: dict[tuple[str, str], float] = {}
        for e in self.edges:
            if e.src in relay_block or e.dst in relay_block:
                continue
            ek = e.key()
            r = min(e.capacity_bps * cfg.dt_s, self.K[ek] / rho)
            if r <= 0:
                continue
            rbar[ek] = r
            psi[ek] = (omega * rho * self.Z[ek]
                       + cfg.bp_beta_prime * e.latency_ms
                       + cfg.bp_chi_prime * self.edge_aos(e, t))
        qsrc = {f.name: float(sum(self.Q[(f.name, f.src)])) for f in self.flows}
        return rbar, psi, qsrc

    def _greedy_plan(self, rbar: dict, psi: dict, qsrc: dict) -> list[tuple]:
        """Per-flow rate-aware greedy over the shared resources (ABLATION).

        Flows are taken in decreasing source backlog.  Each is given the
        path maximising R(P) * W_psi(P) by the bottleneck-restricted
        Dijkstra sweep, subject to a positivity gate on the true path
        weight, and the chosen path's rate is then subtracted from the
        shared residual capacity and key ceilings before the next flow is
        considered.

        The greedy is exact for a single flow and has no bounded-loss
        guarantee for several, because the residual it hands on encodes no
        price for the contention it creates.  `_cg_plan` supplies that
        price through the master LP's duals.
        """
        rho = KEY_BITS_PER_PAYLOAD_BIT
        omega = self.cfg.bp_omega
        resid = dict(rbar)
        plan: list[tuple] = []
        ordered = sorted(self.flows, key=lambda fl: -qsrc.get(fl.name, 0.0))
        for f in ordered:
            q = qsrc.get(f.name, 0.0)
            if q <= 0:
                continue
            cand = {ek: r for ek, r in resid.items() if r > 0}
            if not cand:
                continue
            best = None
            for theta in sorted(set(cand.values())):
                allowed = {ek for ek, r in cand.items() if r >= theta}
                nodes = self._dijkstra_psi(f, allowed, psi)
                if not nodes or len(nodes) < 2:
                    continue
                eks = list(zip(nodes, nodes[1:]))
                if any(ek not in cand for ek in eks):
                    continue
                w_psi = q - sum(psi[ek] for ek in eks)
                w_true = q - omega * rho * sum(self.Z[ek] for ek in eks)
                if w_true <= 0:
                    continue
                r = min(q, min(cand[ek] for ek in eks))
                if r * w_psi > (best[0] if best else 0.0):
                    best = (r * w_psi, eks, r)
            if best is None:
                continue
            _, eks, r = best
            if r <= 0:
                continue
            plan.append((f.name, eks, r))
            for ek in eks:
                resid[ek] -= r
        return plan

    def _solve_master(self, columns: list[dict], rbar: dict,
                      qsrc: dict) -> tuple[list, dict, dict, float]:
        """Solve the restricted master LP and return primal and dual values.

        Variables are path rates in units of LP_SCALE bits and objective
        coefficients are path weights in the same units, so the reported
        objective is the drift reduction divided by LP_SCALE squared.
        Constraint rows are the per-edge ceilings (C1)-(C2), folded into
        one row per edge because both bound the same aggregate, followed
        by the per-flow backlog ceilings (C3).
        """
        if not columns:
            return [], {}, {}, 0.0
        edges_used = sorted({ek for col in columns for ek in col["eks"]})
        eidx = {ek: i for i, ek in enumerate(edges_used)}
        fnames = sorted({col["f"] for col in columns})
        ne = len(edges_used)
        fidx = {fn: ne + i for i, fn in enumerate(fnames)}
        A = np.zeros((ne + len(fnames), len(columns)))
        b = np.empty(ne + len(fnames))
        for j, col in enumerate(columns):
            for ek in col["eks"]:
                A[eidx[ek], j] = 1.0
            A[fidx[col["f"]], j] = 1.0
        for ek, i in eidx.items():
            b[i] = rbar[ek] / LP_SCALE
        for fn, i in fidx.items():
            b[i] = qsrc[fn] / LP_SCALE
        c = np.array([-col["w"] / LP_SCALE for col in columns])
        res = linprog(c, A_ub=A, b_ub=b, bounds=(0.0, None), method="highs")
        if not res.success:
            return [0.0] * len(columns), {}, {}, 0.0
        marg = res.ineqlin.marginals
        # scipy reports duals of `A_ub x <= b_ub` for a MINIMISATION as
        # non-positive numbers; the prices of the maximisation are their
        # negatives.  Clamp at zero to absorb solver noise, which also
        # keeps the pricing costs non-negative so Dijkstra stays valid.
        pi = {ek: max(0.0, -float(marg[i])) for ek, i in eidx.items()}
        sigma = {fn: max(0.0, -float(marg[i])) for fn, i in fidx.items()}
        return list(res.x), pi, sigma, -float(res.fun)

    def _cg_plan(self, rbar: dict, psi: dict,
                 qsrc: dict) -> tuple[list[tuple], dict]:
        """Solve the per-slot max-weight problem EXACTLY by column generation.

        The problem is

            max  sum_{f,P} r_{f,P} ( Q^f_src - sum_{e in P} psi_e )
            s.t. sum_{f, P ni e} r_{f,P} <= min{C_e dt, K_e/rho}   for all e
                 sum_P r_{f,P} <= Q^f_src                          for all f
                 r >= 0,

        a linear program with one column per (flow, simple path).  Given
        edge prices pi_e >= 0 and flow prices sigma_f >= 0 from the
        restricted master, the reduced cost of column (f,P) is

            Q^f_src - sigma_f - sum_{e in P} ( psi_e + pi_e ),

        so the pricing subproblem is a shortest-path problem on the
        non-negative cost psi_e + pi_e: one Dijkstra run per flow.  When no
        column prices out, the restricted master's optimum is the optimum
        of the full LP, and the loop stops.

        This is the step the per-flow greedy is missing.  pi_e is exactly
        the price of the contention a flow creates on edge e, and it is
        what the greedy's residual-capacity bookkeeping fails to charge.
        """
        cfg = self.cfg
        flows_by_name = {f.name: f for f in self.flows}
        active = [f for f in self.flows if qsrc.get(f.name, 0.0) > 0]

        def make_col(fname: str, eks: tuple) -> dict:
            return {"f": fname, "eks": eks,
                    "w": qsrc[fname] - sum(psi[ek] for ek in eks)}

        columns: list[dict] = []
        seen: set[tuple] = set()

        def add(fname: str, eks: tuple) -> bool:
            key = (fname, eks)
            if key in seen or not eks:
                return False
            if any(ek not in rbar for ek in eks):
                return False
            seen.add(key)
            columns.append(make_col(fname, eks))
            return True

        # Warm start: last slots' columns, then the min-psi path.
        for f in active:
            for eks in self.col_pool.get(f.name, []):
                add(f.name, eks)
            nodes = self._dijkstra_psi(f, set(rbar), psi)
            if len(nodes) >= 2:
                add(f.name, tuple(zip(nodes, nodes[1:])))

        x: list[float] = []
        iters = 0
        converged = True
        if columns:
            for iters in range(1, cfg.cg_max_iters + 1):
                x, pi, sigma, _ = self._solve_master(columns, rbar, qsrc)
                cost = {ek: psi[ek] / LP_SCALE + pi.get(ek, 0.0)
                        for ek in rbar}
                added = False
                for f in active:
                    nodes = self._dijkstra_psi(f, set(rbar), cost)
                    if len(nodes) < 2:
                        continue
                    eks = tuple(zip(nodes, nodes[1:]))
                    red = (qsrc[f.name] / LP_SCALE - sigma.get(f.name, 0.0)
                           - sum(cost[ek] for ek in eks))
                    if red > cfg.cg_tol and add(f.name, eks):
                        added = True
                if not added:
                    break
            else:
                converged = False
                self.cg_nonconverged += 1
            x, _, _, obj = self._solve_master(columns, rbar, qsrc)
        else:
            obj = 0.0

        # Refresh the warm-start pool with the columns the LP actually used,
        # most recently useful first, capped so it cannot grow without bound.
        used: dict[str, list[tuple]] = defaultdict(list)
        plan: list[tuple] = []
        for xj, col in zip(x, columns):
            r = xj * LP_SCALE
            if r > 1e-6:
                plan.append((col["f"], list(col["eks"]), r))
                used[col["f"]].append(col["eks"])
        for f in active:
            keep = used[f.name] + [p for p in self.col_pool.get(f.name, [])
                                   if p not in used[f.name]]
            self.col_pool[f.name] = keep[:cfg.cg_pool_per_flow]

        greedy_obj = 0.0
        for fname, eks, r in self._greedy_plan(rbar, psi, qsrc):
            greedy_obj += r * (qsrc[fname] - sum(psi[ek] for ek in eks))
        diag = dict(iters=iters, columns=len(columns),
                    converged=1 if converged else 0,
                    obj_lp=obj * LP_SCALE * LP_SCALE, obj_greedy=greedy_obj)
        return plan, diag

    def _apply_plan(self, plan: list[tuple], t: int, rbar: dict,
                    qsrc: dict) -> dict:
        """Commit a per-slot action: drain queues, spend keys, log state."""
        rho = KEY_BITS_PER_PAYLOAD_BIT
        cfg = self.cfg
        bits_served = 0.0
        rekey_events = 0
        aos_observed: list[float] = []
        served_by_flow: dict[str, float] = defaultdict(float)
        edges_touched: set[tuple[str, str]] = set()
        for fname, eks, r in plan:
            served_by_flow[fname] += r
            bits_served += r
            key_bits = r * rho
            for ek in eks:
                e = self.edge_index[ek]
                self.K[ek] = max(0.0, self.K[ek] - key_bits)
                self._D_cycle[ek] += r
                edges_touched.add(ek)
                if self.K[ek] < e.k_min_bits:
                    rekey_events += 1
                aos_observed.append(self.edge_aos(e, t))
        for f in self.flows:
            remaining = served_by_flow.get(f.name, 0.0)
            dq = self.Q[(f.name, f.src)]
            while remaining > 1e-9 and dq:
                head = dq[0]
                if head <= remaining:
                    remaining -= head
                    dq.popleft()
                else:
                    dq[0] -= remaining
                    remaining = 0.0
        # A secrecy outage is a flow-slot left with backlog while the
        # binding ceiling on the edges it used was the key pool rather than
        # the classical capacity.
        violations = 0
        for fname, total in served_by_flow.items():
            if total >= qsrc.get(fname, 0.0) - 1e-9:
                continue
            eks = [ek for fn, ekl, _ in plan if fn == fname for ek in ekl]
            if not eks:
                continue
            key_bound = min(rbar[ek] for ek in eks)
            cap_bound = min(self.edge_index[ek].capacity_bps * cfg.dt_s
                            for ek in eks)
            if key_bound < cap_bound:
                violations += 1
        # Edges whose ceiling this slot came from the physical pool and that
        # the action saturated: the physical constraint, not the virtual
        # queue, was what bound.
        blocked = sum(
            1 for ek in edges_touched
            if rbar[ek] < self.edge_index[ek].capacity_bps * cfg.dt_s - 1e-6
            and self._D_cycle[ek] >= rbar[ek] - 1e-6)
        return dict(bits_delivered=bits_served, rekey_events=rekey_events,
                    aos_observed=aos_observed, violations=violations,
                    key_blocked_edges=blocked)

    def _step_aos_cg(self, t: int, relay_block: set[str]) -> dict:
        """AoS-BP: exact per-slot max-weight by column generation."""
        rbar, psi, qsrc = self._slot_state(t, relay_block)
        plan, diag = self._cg_plan(rbar, psi, qsrc)
        out = self._apply_plan(plan, t, rbar, qsrc)
        out.update(diag)
        return out

    def _step_alg1_prime(self, t: int, relay_block: set[str]) -> dict:
        """AoS-BP-G: the per-flow rate-aware greedy (ABLATION).

        It keeps the three features that distinguish it from the
        conference heuristic -- the drift-derived edge cost psi_e =
        omega*rho*Z_e + beta*L_e + chi*AoS_e, rate-aware selection by
        maximising R(P)*W_psi(P) instead of minimising an additive cost,
        and a positivity gate -- and it tracks residual capacity across
        flows, so its action is feasible.

        What it does NOT have is a bounded-loss guarantee.  Flows are
        coupled through the shared key ceiling K_e/rho as well as through
        capacity, and handing on a decremented residual charges the next
        flow nothing for the contention the previous one created.  Two
        flows sharing one key-limited edge suffice to make the loss
        against the joint optimum unbounded.  `_cg_plan` prices that
        contention explicitly and is exact; this routine is kept to
        measure what the pricing is worth.
        """
        rbar, psi, qsrc = self._slot_state(t, relay_block)
        plan = self._greedy_plan(rbar, psi, qsrc)
        out = self._apply_plan(plan, t, rbar, qsrc)
        obj = sum(r * (qsrc[fn] - sum(psi[ek] for ek in eks))
                  for fn, eks, r in plan)
        out.update(iters=0, columns=len(plan), converged=1,
                   obj_lp=0.0, obj_greedy=obj)
        return out

    # -----------------------------------------------------------------
    def _route(self, f: Flow, t: int, blocked: set[str]) -> list[str]:
        """Compute a route for flow f under the active scheduler.

        Every scheduler is implemented as a Dijkstra shortest-path over a
        per-edge weight function specific to that scheduler.  This keeps
        the comparison apples-to-apples: the only thing that varies
        across baselines is the edge weight.
        """
        cfg = self.cfg
        sched = cfg.scheduler

        def edge_weight(e: Edge) -> float | None:
            if e.dst in blocked or e.src in blocked:
                return None
            k = self.K[e.key()]
            if sched == "pqc_only" and e.qkd_capable and e.pqc_refresh_bps == 0:
                return None
            if sched == "qkd_only" and not e.qkd_capable:
                return None
            if k < e.k_min_bits:    # reserve constraint
                if sched in ("aos_backpressure", "key_rate_aware"):
                    # Heavy penalty rather than hard reject so a route exists
                    return 1e6 + e.latency_ms
                return None
            if sched == "shortest_path":
                return 1.0
            if sched == "max_throughput":
                return e.latency_ms
            if sched == "pqc_only":
                return e.latency_ms
            if sched == "qkd_only":
                return e.latency_ms
            if sched == "key_rate_aware":
                # Prefer high-key edges: weight inversely proportional to K
                return 1.0 + 1e6 / (k + 1.0)
            if sched == "aos_backpressure":
                # Eq. (8): latency + freshness + key-scarcity gradient.
                return (cfg.bp_beta * e.latency_ms
                        + cfg.bp_chi * self.edge_aos(e, t)
                        + self._key_term(e, k))
            return e.latency_ms

        # Dijkstra
        dist = {n: math.inf for n in self.nodes}
        prev = {n: None for n in self.nodes}
        dist[f.src] = 0.0
        unvisited = set(self.nodes)
        while unvisited:
            u = min(unvisited, key=lambda x: (dist[x], x))
            if dist[u] == math.inf:
                break
            if u == f.dst:
                break
            unvisited.remove(u)
            for e in self.out_edges.get(u, []):
                if e.dst not in unvisited:
                    continue
                w = edge_weight(e)
                if w is None:
                    continue
                d = dist[u] + w
                if d < dist[e.dst]:
                    dist[e.dst] = d
                    prev[e.dst] = u
        return path(prev, f.src, f.dst)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(cfg: SimConfig, qkd_schedule, out_dir: Path):
    nodes, edges = default_topology()
    if cfg.k_max_bits is not None:
        for e in edges:
            e.k_max_bits = cfg.k_max_bits
    flows = default_flows()
    net = AoSNetwork(nodes, edges, flows, cfg, qkd_schedule)
    for t in range(0, cfg.horizon_s, cfg.dt_s):
        net.step(t)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{cfg.scheduler}_{cfg.scenario}_s{cfg.seed}.csv"
    with open(out_dir / fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "secure_bps", "violations", "k_min", "k_mean",
                         "q_total", "aos_mean", "aos_max", "rekey",
                         "q_sumsq", "z_l1", "z_sumsq", "z_max", "lyapunov",
                         "key_inflow", "key_overflow",
                         "cg_iters", "cg_columns", "cg_converged",
                         "obj_lp", "obj_greedy", "key_blocked_edges"])
        for log in net.logs:
            writer.writerow([log.t, log.total_secure_bits, log.total_violations,
                             log.keypool_min_bits, log.keypool_mean_bits,
                             log.queue_total_bits, log.aos_mean,
                             log.aos_max, log.rekey_events,
                             log.q_sumsq, log.z_l1, log.z_sumsq, log.z_max,
                             log.lyapunov, log.key_inflow_bits,
                             log.key_overflow_bits,
                             log.cg_iters, log.cg_columns, log.cg_converged,
                             log.obj_lp, log.obj_greedy,
                             log.key_blocked_edges])
    # Aggregate
    bits = sum(l.total_secure_bits for l in net.logs)
    viols = sum(l.total_violations for l in net.logs)
    rekeys = sum(l.rekey_events for l in net.logs)
    qmean = float(np.mean([l.queue_total_bits for l in net.logs]))
    aosmean = float(np.mean([l.aos_mean for l in net.logs]))
    aosmax = float(np.max([l.aos_max for l in net.logs]))
    n_flow_steps = cfg.horizon_s * len(flows)
    # Strong stability bounds the time-average L1 norms, so those are the
    # quantities to report; the Lyapunov value itself is a diagnostic.
    return dict(
        scheduler=cfg.scheduler,
        scenario=cfg.scenario,
        seed=cfg.seed,
        horizon_s=cfg.horizon_s,
        secure_goodput_bps=bits / cfg.horizon_s,
        secrecy_outage_rate=viols / max(1, n_flow_steps),
        rekey_events=rekeys,
        mean_queue=qmean,
        mean_aos=aosmean,
        max_aos=aosmax,
        mean_z_l1=float(np.mean([l.z_l1 for l in net.logs])),
        final_z_l1=float(net.logs[-1].z_l1),
        max_z_edge=float(np.max([l.z_max for l in net.logs])),
        mean_lyapunov=float(np.mean([l.lyapunov for l in net.logs])),
        final_lyapunov=float(net.logs[-1].lyapunov),
        key_inflow_bits=float(sum(l.key_inflow_bits for l in net.logs)),
        key_overflow_bits=float(sum(l.key_overflow_bits for l in net.logs)),
        key_overflow_frac=float(sum(l.key_overflow_bits for l in net.logs)
                                / max(1.0, sum(l.key_inflow_bits
                                               for l in net.logs))),
        # Overflow restricted to edges that carried traffic: the well-posed
        # test of Theorem 1 assumption (iii).
        # Of the key material actually produced on traffic-carrying edges,
        # what fraction was discarded at the cap.  This is the well-posed
        # form of Theorem 2 assumption (iii): with gating, material that is
        # never generated is not "lost", so the denominator is generation,
        # not capability.
        active_overflow_frac=float(
            sum(net.cum_overflow[k] for k in net.cum_D if net.cum_D[k] > 0)
            / max(1.0, sum(net.cum_generated[k] for k in net.cum_D
                           if net.cum_D[k] > 0))),
        pqc_utilisation=float(
            sum(net.cum_generated.values())
            / max(1.0, sum(net.cum_G.values()))),
        n_active_edges=int(sum(1 for k in net.cum_D if net.cum_D[k] > 0)),
        weight_variant=cfg.weight_variant,
        k_max_bits=(cfg.k_max_bits if cfg.k_max_bits is not None
                    else edges[0].k_max_bits),
        aos_T_a=cfg.aos_T_a,
        load_scale=cfg.load_scale,
        # Column-generation audit.  `cg_gap_frac` is the fraction of the
        # exact per-slot max-weight objective that the per-flow greedy
        # forgoes, averaged over the slots in which the optimum is
        # positive.  `cg_nonconverged` counts slots that hit the iteration
        # cap before the pricing step certified optimality; it must be 0
        # for the exactness claim to hold.
        cg_mean_iters=float(np.mean([l.cg_iters for l in net.logs])),
        cg_max_iters_used=int(max([l.cg_iters for l in net.logs] or [0])),
        cg_nonconverged=int(net.cg_nonconverged),
        cg_gap_frac=float(np.mean(
            [1.0 - l.obj_greedy / l.obj_lp for l in net.logs if l.obj_lp > 0]
            or [0.0])),
        cg_gap_max=float(np.max(
            [1.0 - l.obj_greedy / l.obj_lp for l in net.logs if l.obj_lp > 0]
            or [0.0])),
        key_blocked_edge_slots=int(sum(l.key_blocked_edges
                                       for l in net.logs)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--schedulers", nargs="+",
                    default=["shortest_path", "pqc_only", "qkd_only",
                             "key_rate_aware", "aos_backpressure",
                             "aos_greedy", "aos_cg", "aos_ideal"])
    ap.add_argument("--scenarios", nargs="+",
                    default=["nominal", "weather", "relay_compromise",
                             "traffic_surge", "coalition_partition"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--weight-variant", default="W3",
                    choices=["W0", "W1", "W2", "W3"],
                    help="form of the Eq. (8) key-scarcity term")
    ap.add_argument("--k-max", type=float, default=None,
                    help="override every edge's key-buffer cap, in bits")
    ap.add_argument("--aos-ta", type=float, default=300.0,
                    help="Eq. (2) cap T_a on the age term, seconds")
    ap.add_argument("--load-scale", type=float, default=1.0)
    ap.add_argument("--start-jd", type=float, default=None,
                    help="TT Julian date to anchor SGP4 propagation at. "
                         "Defaults to the median TLE epoch of the snapshot, "
                         "which is what reproduces its actual orbital state.")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    out_dir = (here.parent / args.out).resolve()

    # Persist provenance once for the paper.
    here = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Precomputing per-seed QKD pass schedules (each pass schedule "
          "uses a different per-pass cloud-transmittance realization)")
    schedules: dict[int, dict] = {}
    provenance_str = ""
    for seed in args.seeds:
        t0 = time.time()
        sched, prov = build_qkd_schedule(start_jd=args.start_jd, hours=12,
                                         weather_seed=seed)
        schedules[seed] = sched
        provenance_str = prov
        print(f"  seed {seed}: {time.time()-t0:.1f} s")
    print(f"  constellation: {provenance_str}")
    (out_dir / "provenance.txt").write_text(provenance_str + "\n")

    rows = []
    for sched in args.schedulers:
        for sc in args.scenarios:
            for seed in args.seeds:
                cfg = SimConfig(horizon_s=args.horizon, seed=seed,
                                scheduler=sched, scenario=sc,
                                weight_variant=args.weight_variant,
                                k_max_bits=args.k_max,
                                aos_T_a=args.aos_ta,
                                load_scale=args.load_scale)
                r = run(cfg, schedules[seed], out_dir)
                rows.append(r)
                print(f"[done] {sched:>18s}  {sc:>20s}  s={seed}  "
                      f"goodput={r['secure_goodput_bps']/1e6:.2f}Mbps  "
                      f"outage={r['secrecy_outage_rate']:.3f}  "
                      f"aos_mean={r['mean_aos']:.2f}")
    master = out_dir / "master.csv"
    with open(master, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {master}")


if __name__ == "__main__":
    main()
