"""Hybrid-supply NTN simulator: routing and key manufacturing jointly.

This implements the model and the policy of the journal manuscript.  It
supersedes `aos_network.py`, which modelled a single fungible key pool
and a single queue per link; both of those turned out to be the wrong
abstraction.

WHAT IS MODELLED

Two species of key material, which are not interchangeable.

  Q-keys  arrive exogenously from satellite passes and are
          information-theoretically secure.
  P-keys  are MANUFACTURED by running ML-KEM under a per-node
          encapsulation budget, and are computationally secure.

A delivered block is information-theoretically graded only if EVERY hop
on its route encrypted it with a Q-key, so the two species are
complements rather than substitutes: no compute budget moves a single
block of graded traffic.

Each edge is split three ways, after the tandem-queue decomposition of
Akhtar et al. (IEEE/ACM ToN 31(5):2281-2296, 2023) extended to two
species:

  Xq[e]  blocks waiting for a Q-key      served at the Q-key rate
  Xp[e]  blocks waiting for a P-key      served at the manufactured rate
  Y[e]   encrypted blocks waiting        served at the link rate C_e

Routing is a minimum-weight path on

  W_e = Yt[e] + min{ Xqt[e] + V.chi*age_e , Xpt[e] + V.chi*T_a }   flexible
  W_e = Yt[e] +      Xqt[e] + V.chi*age_e                          graded

over the virtual (precedence-relaxed) queues, and the inner minimum is
the species decision.  Manufacturing is a per-node continuous knapsack
on (Xpt[e] - V.nu*c_e)/c_e, which is demand gating derived rather than
imposed.

UNITS.  Everything is in BLOCKS.  One block is 8 kB of payload under one
256-bit session key, so one key unit encrypts one block and no
conversion constant appears in the scheduler.

Author: Liang Dong.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from aos_network import (DEFAULT_GROUND_STATIONS, Edge, build_qkd_schedule,
                         default_topology, qkd_rate_at)

# --- unit system -----------------------------------------------------------
PAYLOAD_BITS_PER_BLOCK = 8 * 1024 * 8      # 8 kB of payload
KEY_BITS_PER_UNIT = 256                    # one AES-256 session key
# One ML-KEM-768 encapsulation yields one shared secret, hence one key
# unit, so manufacturing cost is one operation per unit on every edge.
OPS_PER_KEY_UNIT = 1.0


def bps_to_blocks(bps: float, dt: float) -> float:
    return bps * dt / PAYLOAD_BITS_PER_BLOCK


def keybps_to_units(bps: float, dt: float) -> float:
    return bps * dt / KEY_BITS_PER_UNIT


# Nominal offered load, in Mbps, aggregated over the five flows and kept
# in their original proportions.  It is set from the capacity region
# rather than inherited: the linear program of `region_boundary` puts the
# boundary at 36.5 Mbps for theta = 1/2 with the node budget below, and
# the nominal point sits at 0.68 of it.  The conference and earlier
# journal versions offered 245 Mbps, which is outside this region by a
# factor of seven; they could do so only because every edge was given its
# own post-quantum refresh and the two key species were pooled.
NOMINAL_TOTAL_MBPS = 25.0


def scaled_flows(total_mbps: float = NOMINAL_TOTAL_MBPS):
    """The five reference flows, rescaled to a given aggregate rate."""
    from aos_network import default_flows
    fl = default_flows()
    s = total_mbps * 1e6 / sum(f.arrival_bps for f in fl)
    for f in fl:
        f.arrival_bps *= s
    return fl


@dataclasses.dataclass
class TqdConfig:
    horizon_s: int = 600
    dt_s: float = 1.0
    seed: int = 0
    scenario: str = "nominal"
    load_scale: float = 1.0
    # Fraction of every class's rate that requires information-theoretic
    # keys.  theta = 1 with budget 0 is the prior work's setting.
    theta: float = 0.5
    # Per-node encapsulation budget, in key units per slot.  One core
    # devoted to ML-KEM-768 at the liboqs throughput point, network
    # overhead included, is about 200 kbps of key material.  Note this
    # is a NODE budget shared across the node's outgoing edges; the
    # previous model gave every edge its own, which silently assumed one
    # core per edge.
    node_budget_units: float = 200_000.0 / KEY_BITS_PER_UNIT
    # Per-edge ceiling on manufacture, same throughput point.
    edge_h_max_units: float = 200_000.0 / KEY_BITS_PER_UNIT
    T_a: float = 300.0                 # key expiry and AoS cap, slots
    # Freshness weight.  Queues are in blocks, so V_chi*T_a must be
    # commensurable with a virtual queue length; Section VI selects it.
    V_chi: float = 10.0
    V_nu: float = 0.0                  # manufacturing cost weight
    policy: str = "aos_tqd"            # aos_tqd | tqd | fungible | sp
    manufacture: bool = True


# ---------------------------------------------------------------------------
# key bank: freshest-first consumption with expiry
# ---------------------------------------------------------------------------
class KeyBank:
    """Units held with their generation slot, consumed freshest first.

    Freshest-first is a policy choice, not physics.  It keeps the age of
    the material actually consumed small, which is what AoS measures,
    and the expiry rule bounds it by T_a, which is what makes the
    available-key process bounded and the drift constant finite.
    """

    __slots__ = ("units", "T_a")

    def __init__(self, T_a: float):
        self.units: deque = deque()      # (gen_slot, amount), oldest first
        self.T_a = T_a

    def deposit(self, t: int, amount: float, species: str = "Q") -> None:
        if amount <= 0:
            return
        if self.units and self.units[-1][0] == t \
                and self.units[-1][2] == species:
            g, a, sp = self.units[-1]
            self.units[-1] = (g, a + amount, sp)
        else:
            self.units.append((t, amount, species))

    def expire(self, t: int) -> float:
        dropped = 0.0
        while self.units and (t - self.units[0][0]) > self.T_a:
            dropped += self.units.popleft()[1]
        return dropped

    def available(self) -> float:
        return sum(u[1] for u in self.units)

    def oldest_age(self, t: int) -> float:
        return (t - self.units[0][0]) if self.units else 0.0

    def mean_age(self, t: int) -> float:
        """Unit-weighted mean age of the bank.

        This is the surrogate the routing weight uses for alpha_e, the
        age of the material a block will actually consume.  The age of
        the OLDEST unit is a poor surrogate under freshest-first
        consumption, because a stale unit simply sits at the bottom of
        the bank until it expires and pins the surrogate at T_a whatever
        the edge is really spending.  The mean moves with both the
        recency of supply and the amount of old material held, which is
        what the routing decision needs to see.
        """
        tot = sum(u[1] for u in self.units)
        if tot <= 0:
            return 0.0
        return sum((t - g) * a for g, a, _ in self.units) / tot

    def consume(self, t: int, amount: float) -> tuple[float, float]:
        """Take `amount` units, freshest first.

        Returns (taken, age_of_oldest_unit_taken, taken_that_were_P).
        The second value is alpha_e(t) of the manuscript: the age of the
        oldest unit that actually encrypted a block this slot.  The
        third exists so that a policy which pools the two species can be
        audited against what it actually spent.
        """
        taken, oldest, from_p = 0.0, 0.0, 0.0
        while amount > 1e-12 and self.units:
            g, a, sp = self.units[-1]
            use = min(a, amount)
            taken += use
            amount -= use
            oldest = max(oldest, t - g)
            if sp == "P":
                from_p += use
            if use >= a - 1e-12:
                self.units.pop()
            else:
                self.units[-1] = (g, a - use, sp)
        return taken, oldest, from_p


# ---------------------------------------------------------------------------
# a batch of blocks travelling together on one marked route
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Batch:
    cls: str
    graded: bool
    route: tuple                 # ((edge_key, species), ...)
    hop: int                     # index of the next hop to traverse
    amount: float
    born: int
    worst_age: float = 0.0       # max alpha over the hops encrypted so far
    used_p: bool = False         # any hop encrypted with a P-key


class HybridNetwork:
    def __init__(self, nodes, edges, flows, cfg: TqdConfig, qkd_schedule):
        self.nodes, self.edges, self.flows = nodes, edges, flows
        self.cfg, self.qkd_schedule = cfg, qkd_schedule
        self.eidx = {e.key(): e for e in edges}
        self.out_edges = defaultdict(list)
        for e in edges:
            self.out_edges[e.src].append(e)

        ek = [e.key() for e in edges]
        self.C = {k: bps_to_blocks(self.eidx[k].capacity_bps, cfg.dt_s)
                  for k in ek}
        self.Xq = {k: deque() for k in ek}
        self.Xp = {k: deque() for k in ek}
        self.Y = {k: deque() for k in ek}
        self.bq = {k: KeyBank(cfg.T_a) for k in ek}
        self.bp = {k: KeyBank(cfg.T_a) for k in ek}
        self.Xqt = {k: 0.0 for k in ek}
        self.Xpt = {k: 0.0 for k in ek}
        self.Yt = {k: 0.0 for k in ek}

        self.log: list[dict] = []
        # optional per-edge trace, for the mechanism figure
        self.trace_edge: tuple | None = None
        self.trace: list[dict] = []
        self.delivered: list[tuple] = []      # (t, cls, graded, aos, delay)
        self.cum_servedQ = {k: 0.0 for k in ek}
        self.cum_servedP = {k: 0.0 for k in ek}
        self.cum_manufactured = 0.0
        self.cum_capability = 0.0
        self.cum_qkd = 0.0
        self.cum_expiredQ = 0.0
        self.blocked = 0

    # -- routing ----------------------------------------------------------
    def _weights(self, t: int, blocked: set[str]):
        """Edge weights of Eq. (12), one map per class type.

        Returns (Wq, Wf, spec) where Wq is the graded weight, Wf the
        flexible weight, and spec[e] the species attaining the flexible
        minimum.
        """
        cfg = self.cfg
        Wq, Wf, spec = {}, {}, {}
        for e in self.edges:
            k = e.key()
            if e.src in blocked or e.dst in blocked:
                continue
            age = self.bq[k].mean_age(t)
            cq = self.Xqt[k] + cfg.V_chi * min(age, cfg.T_a)
            cp = self.Xpt[k] + cfg.V_chi * cfg.T_a
            Wq[k] = self.Yt[k] + cq
            if cp < cq:
                Wf[k], spec[k] = self.Yt[k] + cp, "P"
            else:
                Wf[k], spec[k] = self.Yt[k] + cq, "Q"
        return Wq, Wf, spec

    def _dijkstra(self, src: str, dst: str, W: dict) -> list | None:
        dist = {n: math.inf for n in self.nodes}
        prev = {n: None for n in self.nodes}
        dist[src] = 0.0
        unvis = set(self.nodes)
        while unvis:
            u = min(unvis, key=lambda x: (dist[x], x))
            if dist[u] == math.inf:
                return None
            if u == dst:
                break
            unvis.remove(u)
            for e in self.out_edges.get(u, []):
                k = e.key()
                if k not in W or e.dst not in unvis:
                    continue
                d = dist[u] + W[k]
                if d < dist[e.dst]:
                    dist[e.dst], prev[e.dst] = d, u
        if dist[dst] == math.inf:
            return None
        seq, cur = [], dst
        while cur is not None:
            seq.append(cur)
            if cur == src:
                break
            cur = prev[cur]
        seq.reverse()
        return seq if seq and seq[0] == src else None

    # -- manufacturing ----------------------------------------------------
    def _manufacture(self) -> dict:
        """Proposition 2: per-node continuous knapsack on (Xpt - V.nu.c)/c.

        Edges whose waiting backlog does not cover the marginal cost of
        an encapsulation are skipped, which is exactly demand gating.
        """
        cfg = self.cfg
        h = {e.key(): 0.0 for e in self.edges}
        if not cfg.manufacture:
            return h
        for n in self.nodes:
            outs = [e.key() for e in self.out_edges.get(n, [])]
            if not outs:
                continue
            w = {k: self.Xpt[k] - cfg.V_nu * OPS_PER_KEY_UNIT for k in outs}
            left = cfg.node_budget_units
            for k in sorted(outs, key=lambda x: -w[x] / OPS_PER_KEY_UNIT):
                if left <= 0 or w[k] <= 0:
                    break
                take = min(cfg.edge_h_max_units, left / OPS_PER_KEY_UNIT)
                h[k] = take
                left -= OPS_PER_KEY_UNIT * take
        return h

    # -- one slot ---------------------------------------------------------
    def step(self, t: int) -> None:
        cfg = self.cfg
        blocked, wmult, surge = set(), 1.0, 1.0
        if cfg.scenario == "weather":
            wmult = 0.2 if (t // 120) % 2 == 0 else 1.0
        elif cfg.scenario == "relay_compromise" and 200 <= t < 400:
            blocked.add("GW-EU")
        elif cfg.scenario == "traffic_surge" and 200 <= t < 400:
            surge = 3.0
        elif cfg.scenario == "coalition_partition" and 250 <= t < 500:
            blocked.add("GW-APAC")

        Wq, Wf, spec = self._weights(t, blocked)

        # 1. route arrivals -----------------------------------------------
        aq = defaultdict(float); ap = defaultdict(float)
        for f in self.flows:
            total = bps_to_blocks(f.arrival_bps, cfg.dt_s) * cfg.load_scale * surge
            for graded, share in ((True, cfg.theta), (False, 1.0 - cfg.theta)):
                amt = total * share
                if amt <= 0:
                    continue
                W = Wq if graded else Wf
                seq = self._dijkstra(f.src, f.dst, W)
                if seq is None or len(seq) < 2:
                    self.blocked += 1
                    continue
                eks = list(zip(seq, seq[1:]))
                route = tuple((k, "Q" if graded else spec[k]) for k in eks)
                for k, sp in route:
                    (aq if sp == "Q" else ap)[k] += amt
                b = Batch(f.name, graded, route, 0, amt, t)
                first_k, first_sp = route[0]
                (self.Xq if first_sp == "Q" else self.Xp)[first_k].append(b)

        # 2. manufacture ---------------------------------------------------
        h = self._manufacture()

        # 3. supply --------------------------------------------------------
        gs_names = {g.name for g in DEFAULT_GROUND_STATIONS}
        qkd_units = defaultdict(float)
        for gs in gs_names:
            rate = qkd_rate_at(t, self.qkd_schedule.get(gs, [])) * wmult
            if rate <= 0:
                continue
            for e in self.edges:
                if e.qkd_capable and (e.src == gs or e.dst == gs):
                    qkd_units[e.key()] += keybps_to_units(rate, cfg.dt_s)
        for e in self.edges:
            k = e.key()
            self.bq[k].deposit(t, qkd_units[k])
            self.cum_qkd += qkd_units[k]
            if cfg.policy == "fungible":
                # a single pooled species: manufactured material is
                # deposited into the same bank and counts as if it were
                # information-theoretic, which is the error this paper
                # is about
                self.bq[k].deposit(t, h[k], "P")
            else:
                self.bp[k].deposit(t, h[k], "P")
            self.cum_manufactured += h[k]
            self.cum_capability += min(cfg.edge_h_max_units,
                                       cfg.node_budget_units)
            self.cum_expiredQ += self.bq[k].expire(t)
            self.bp[k].expire(t)

        # 4. encrypt --------------------------------------------------------
        dq = defaultdict(float); dp = defaultdict(float)
        for k in self.C:
            for queue, bank, served in ((self.Xq[k], self.bq[k], dq),
                                        (self.Xp[k], self.bp[k], dp)):
                budget = bank.available()
                while queue and budget > 1e-9:
                    b = queue[0]
                    use = min(b.amount, budget)
                    got, age, from_p = bank.consume(t, use)
                    if got <= 1e-12:
                        break
                    budget -= got
                    served[k] += got
                    # a block is computationally graded if ANY hop spent
                    # manufactured material, whatever the policy believed
                    moved = Batch(b.cls, b.graded, b.route, b.hop, got,
                                  b.born, max(b.worst_age, age),
                                  b.used_p or (b.route[b.hop][1] == "P")
                                  or from_p > 1e-9)
                    self.Y[k].append(moved)
                    if got >= b.amount - 1e-12:
                        queue.popleft()
                    else:
                        b.amount -= got

        for k in self.C:
            self.cum_servedQ[k] += dq[k]
            self.cum_servedP[k] += dp[k]

        # 5. forward --------------------------------------------------------
        for k in self.C:
            budget = self.C[k]
            q = self.Y[k]
            # nearest to origin first
            ordered = sorted(q, key=lambda b: b.hop)
            q.clear()
            for b in ordered:
                if budget <= 1e-9:
                    q.append(b)
                    continue
                use = min(b.amount, budget)
                budget -= use
                nxt = b.hop + 1
                if nxt >= len(b.route):
                    aos = self.cfg.T_a if b.used_p else min(b.worst_age,
                                                            self.cfg.T_a)
                    self.delivered.append((t, b.cls, not b.used_p, aos,
                                           t - b.born, use, b.graded))
                else:
                    nk, nsp = b.route[nxt]
                    fwd = Batch(b.cls, b.graded, b.route, nxt, use, b.born,
                                b.worst_age, b.used_p)
                    (self.Xq if nsp == "Q" else self.Xp)[nk].append(fwd)
                if use < b.amount - 1e-12:
                    b.amount -= use
                    q.append(b)

        # 6. virtual queues --------------------------------------------------
        for k in self.C:
            self.Xqt[k] = max(0.0, self.Xqt[k] + aq[k] - qkd_units[k])
            self.Xpt[k] = max(0.0, self.Xpt[k] + ap[k] - h[k])
            self.Yt[k] = max(0.0, self.Yt[k] + aq[k] + ap[k] - self.C[k])

        if self.trace_edge is not None:
            k = self.trace_edge
            self.trace.append(dict(
                t=t,
                qkd=qkd_units[k], mfg=h[k],
                bankQ=self.bq[k].available(), bankP=self.bp[k].available(),
                xq=sum(b.amount for b in self.Xq[k]),
                xp=sum(b.amount for b in self.Xp[k]),
                y=sum(b.amount for b in self.Y[k]),
                servedQ=dq[k], servedP=dp[k],
                ageQ=self.bq[k].mean_age(t),
                ageOldest=self.bq[k].oldest_age(t)))

        phys = (sum(sum(b.amount for b in self.Xq[k]) for k in self.C)
                + sum(sum(b.amount for b in self.Xp[k]) for k in self.C)
                + sum(sum(b.amount for b in self.Y[k]) for k in self.C))
        self.log.append(dict(
            t=t,
            virt=sum(self.Xqt.values()) + sum(self.Xpt.values())
                 + sum(self.Yt.values()),
            phys=phys,
            manufactured=sum(h.values()),
            lyap=sum(self.Xqt[k] ** 2 + self.Xpt[k] ** 2 + self.Yt[k] ** 2
                     for k in self.C),
        ))


def run(cfg: TqdConfig, qkd_schedule) -> dict:
    nodes, edges = default_topology()
    flows = scaled_flows()
    if cfg.policy == "tqd":                 # prior work: no manufacture,
        cfg = dataclasses.replace(cfg, manufacture=False, theta=1.0)
    net = HybridNetwork(nodes, edges, flows, cfg, qkd_schedule)
    for t in range(cfg.horizon_s):
        net.step(t)

    d = net.delivered
    tot = sum(x[5] for x in d)
    its = sum(x[5] for x in d if x[2])
    claimed = sum(x[5] for x in d if x[6])
    mislabelled = sum(x[5] for x in d if x[6] and not x[2])
    aos = ([x[3] for x in d for _ in range(1)] or [0.0])
    wts = np.array([x[5] for x in d]) if d else np.array([0.0])
    aos_arr = np.array([x[3] for x in d]) if d else np.array([0.0])
    dly = np.array([x[4] for x in d]) if d else np.array([0.0])
    half = len(net.log) // 2
    slope = float(np.polyfit([l["t"] for l in net.log[half:]],
                             [l["phys"] for l in net.log[half:]], 1)[0]) \
        if len(net.log) > 4 else float("nan")
    return dict(
        policy=cfg.policy, scenario=cfg.scenario, seed=cfg.seed,
        theta=cfg.theta, load_scale=cfg.load_scale,
        V_nu=cfg.V_nu, budget=cfg.node_budget_units,
        goodput_mbps=tot * PAYLOAD_BITS_PER_BLOCK / cfg.horizon_s / 1e6,
        its_fraction=its / max(tot, 1e-12),
        claimed_its_fraction=claimed / max(tot, 1e-12),
        # blocks a policy declared information-theoretic that were in
        # fact encrypted with manufactured material somewhere on the way
        mislabelled_fraction=mislabelled / max(claimed, 1e-12),
        mean_aos=float(np.average(aos_arr, weights=wts)) if d else 0.0,
        mean_delay=float(np.average(dly, weights=wts)) if d else 0.0,
        mean_virt=float(np.mean([l["virt"] for l in net.log])),
        mean_phys=float(np.mean([l["phys"] for l in net.log])),
        phys_slope=slope,
        final_lyap=float(net.log[-1]["lyap"]),
        manufactured_units=net.cum_manufactured,
        capability_units=net.cum_capability,
        manufacture_utilisation=net.cum_manufactured
                                / max(net.cum_capability, 1e-12),
        qkd_units=net.cum_qkd,
        qkd_expired_frac=net.cum_expiredQ / max(net.cum_qkd, 1e-12),
        unroutable=net.blocked,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--policies", nargs="+",
                    default=["aos_tqd", "tqd", "fungible"])
    ap.add_argument("--scenarios", nargs="+", default=["nominal"])
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.5])
    ap.add_argument("--out", default="results_tqd")
    args = ap.parse_args()

    out = (Path(__file__).resolve().parent.parent / args.out)
    out.mkdir(parents=True, exist_ok=True)
    sched = {s: build_qkd_schedule(weather_seed=s, hours=12)[0]
             for s in args.seeds}

    rows = []
    for pol in args.policies:
        for sc in args.scenarios:
            for th in args.thetas:
                for s in args.seeds:
                    cfg = TqdConfig(horizon_s=args.horizon, seed=s,
                                    scenario=sc, theta=th, policy=pol)
                    r = run(cfg, sched[s])
                    rows.append(r)
                    print(f"[done] {pol:>9s} {sc:>20s} theta={th:4.2f} "
                          f"s={s} gp={r['goodput_mbps']:7.2f} "
                          f"its={r['its_fraction']:5.3f} "
                          f"aos={r['mean_aos']:7.2f} "
                          f"mfg={r['manufacture_utilisation']:5.3f}")
    with open(out / "master.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", out / "master.csv")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# the capacity region of this instance, by linear programming
# ---------------------------------------------------------------------------
def empirical_eta(schedule, horizon: int, edges, dt: float = 1.0) -> dict:
    """Time-averaged Q-key supply per edge, in key units per slot."""
    gs = {g.name for g in DEFAULT_GROUND_STATIONS}
    tot = defaultdict(float)
    for t in range(horizon):
        for g in gs:
            r = qkd_rate_at(t, schedule.get(g, []))
            if r <= 0:
                continue
            for e in edges:
                if e.qkd_capable and (e.src == g or e.dst == g):
                    tot[e.key()] += keybps_to_units(r, dt)
    return {e.key(): tot[e.key()] / horizon for e in edges}


def _simple_paths(adj, s, d, cutoff=8):
    """All simple s-d paths of at most `cutoff` vertices.

    The cutoff must be large enough that the enumeration is complete for
    the instance, or the linear program silently understates the region.
    On the ten-node topology of Section VI the boundary is unchanged for
    every cutoff at or above six; eight is used throughout.
    """
    out, stack = [], [(s, [s], {s})]
    while stack:
        u, seq, seen = stack.pop()
        if u == d:
            out.append(seq); continue
        if len(seq) > cutoff:
            continue
        for v in adj.get(u, ()):
            if v not in seen:
                stack.append((v, seq + [v], seen | {v}))
    return out


def region_boundary(eta, cfg: TqdConfig, flows, edges, nodes,
                    blocked=frozenset()):
    """Largest t with t*lambda inside Lambda_S, for this instance.

    Solves the linear program of Definition 4 directly: per class, per
    path, per species rates; per-edge manufacturing; per-node budget;
    and the grade restriction that a graded class may not use P-keys.
    Returns the scaling of the nominal arrival vector at the boundary.
    """
    from scipy.optimize import linprog
    eks = [e.key() for e in edges
           if e.src not in blocked and e.dst not in blocked]
    C = {k: bps_to_blocks(dict((e.key(), e) for e in edges)[k].capacity_bps,
                          cfg.dt_s) for k in eks}
    adj = defaultdict(list)
    for (u, v) in eks:
        adj[u].append(v)
    classes = []
    for f in flows:
        lam = bps_to_blocks(f.arrival_bps, cfg.dt_s)
        if cfg.theta > 0:
            classes.append((f.src, f.dst, lam * cfg.theta, True))
        if cfg.theta < 1:
            classes.append((f.src, f.dst, lam * (1 - cfg.theta), False))
    cols = []
    for ci, (s, d, lam, graded) in enumerate(classes):
        for seq in _simple_paths(adj, s, d):
            path = list(zip(seq, seq[1:]))
            if any(k not in C for k in path):
                continue
            for sp in ("Q",) if graded else ("Q", "P"):
                cols.append((ci, path, sp))
    if not cols:
        return 0.0
    eidx = {k: i for i, k in enumerate(eks)}
    nv = len(cols) + len(eks) + 1
    ti = nv - 1
    A, b = [], []
    for k in eks:                                  # classical capacity
        row = np.zeros(nv)
        for j, (_, pth, _) in enumerate(cols):
            if k in pth:
                row[j] = 1.0
        A.append(row); b.append(C[k])
    for k in eks:                                  # Q supply
        row = np.zeros(nv)
        for j, (_, pth, sp) in enumerate(cols):
            if sp == "Q" and k in pth:
                row[j] = 1.0
        A.append(row); b.append(eta.get(k, 0.0))
    for k in eks:                                  # P supply vs manufacture
        row = np.zeros(nv)
        for j, (_, pth, sp) in enumerate(cols):
            if sp == "P" and k in pth:
                row[j] = 1.0
        row[len(cols) + eidx[k]] = -1.0
        A.append(row); b.append(0.0)
    for k in eks:                                  # per-edge ceiling
        row = np.zeros(nv)
        row[len(cols) + eidx[k]] = 1.0
        A.append(row); b.append(cfg.edge_h_max_units if cfg.manufacture
                                else 0.0)
    for n in nodes:                                # per-node budget
        row = np.zeros(nv)
        for k in eks:
            if k[0] == n:
                row[len(cols) + eidx[k]] = OPS_PER_KEY_UNIT
        A.append(row); b.append(cfg.node_budget_units if cfg.manufacture
                                else 0.0)
    Aeq, beq = [], []
    for ci, (_, _, lam, _) in enumerate(classes):
        row = np.zeros(nv)
        for j, (cj, _, _) in enumerate(cols):
            if cj == ci:
                row[j] = 1.0
        row[ti] = -lam
        Aeq.append(row); beq.append(0.0)
    obj = np.zeros(nv); obj[ti] = -1.0
    res = linprog(obj, A_ub=np.array(A), b_ub=np.array(b),
                  A_eq=np.array(Aeq), b_eq=np.array(beq),
                  bounds=[(0, None)] * nv, method="highs")
    return float(res.x[ti]) if res.success else 0.0
