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

A delivered block is QKD-keyed only if EVERY hop on its route encrypted
it with a Q-key, so the two species are not fungible and substitute only
one way: no compute budget moves a single block of graded traffic.

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
the species decision.  age_e is the mean age of the material the edge is
about to SPEND, not of the material it holds.

Manufacturing maximizes sum_e (Xpt[e] - V.nu*c_e) h_e over the budget
polytope.  Because ML-KEM charges both endpoints, that polytope is not a
product over nodes and the closed-form knapsack of the one-sided model
is not optimal; the program is solved exactly.  What survives is the
pruning of edges with nonpositive weight, which is demand gating derived
rather than imposed.

Blocks are served nearest-to-origin first at EVERY arc, internal ones
included.  That is what carries virtual stability to the physical queues
once key material expires, since the pathwise domination the prior work
uses needs keys to be stored forever.

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
# ML-KEM is a two-party mechanism, and FIPS 203 splits the work across
# both endpoints: the responder runs KeyGen and Decaps, the initiator
# runs Encaps.  One key unit on edge (i,j) therefore charges BOTH i and
# j, and the per-node budgets no longer decouple across nodes.  Earlier
# versions of this model charged the tail node only, which understated
# the cost of a key unit by a factor of two and made the manufacturing
# subproblem separable when it is not.
OPS_TAIL = 1.0      # Encaps, at the node the edge leaves
OPS_HEAD = 1.0      # KeyGen + Decaps, at the node the edge enters
OPS_PER_KEY_UNIT = OPS_TAIL + OPS_HEAD      # total KEM work per unit


def qkd_edges_by_station(edges) -> dict:
    """Quantum-capable edges incident on each ground station.

    A station's pass yield is shared across exactly these edges, so this
    map is what keeps one pass from being counted several times.
    """
    gs = {g.name for g in DEFAULT_GROUND_STATIONS}
    out = {g: [] for g in gs}
    for e in edges:
        if not e.qkd_capable:
            continue
        for g in (e.src, e.dst):
            if g in gs:
                out[g].append(e.key())
    return out


def bps_to_blocks(bps: float, dt: float) -> float:
    return bps * dt / PAYLOAD_BITS_PER_BLOCK


def keybps_to_units(bps: float, dt: float) -> float:
    return bps * dt / KEY_BITS_PER_UNIT


# Nominal offered load, in Mbps, aggregated over the five flows and kept
# in their original proportions.  It is set from the capacity region
# rather than inherited: the linear program of `region_boundary` puts the
# boundary at 21.4 Mbps for theta = 1/2 with the node budget below, and
# the nominal point sits at 0.29 of it.
#
# Two corrections moved this number down from the 25 Mbps of the first
# journal draft.  A ground station's pass yield is one key stream and is
# now divided among its four quantum-capable edges instead of being
# credited to each of them in full, which divides eta by four.  And key
# establishment is charged at both endpoints, which roughly halves the
# effective manufacturing budget.  The conference version offered
# 245 Mbps, outside the corrected region by a factor of forty.
NOMINAL_TOTAL_MBPS = 6.25


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
        """Unit-weighted mean age of the whole bank.

        Kept for the traces.  It is NOT the routing surrogate: see
        `marginal_age`, which fixes an ordering defect this measure has.
        """
        tot = sum(u[1] for u in self.units)
        if tot <= 0:
            return 0.0
        return sum((t - g) * a for g, a, _ in self.units) / tot

    def marginal_age(self, t: int, demand: float) -> float:
        """Unit-weighted mean age of the freshest `demand` units.

        This is the surrogate the routing weight uses for alpha_e, and
        it is the age of the material the edge is about to spend rather
        than of the material it happens to hold.  Two earlier choices
        were both wrong, in opposite ways.

        The age of the OLDEST unit is meaningless under freshest-first
        consumption: a stale unit sits at the bottom of the bank until
        it expires and pins the surrogate at T_a whatever the edge is
        really spending.

        The mean over the WHOLE bank can reverse the true ordering.  A
        bank holding ages {0, T_a} has mean T_a/2 but will next spend a
        unit of age 0, while a bank holding {T_a/3, T_a/3} has the lower
        mean T_a/3 and will next spend a unit three times older.  Taking
        the mean over only the units that the current backlog would
        consume removes that inversion, because it looks at the same
        units freshest-first service will actually reach.

        An empty bank returns 0.  Scarcity is already priced by the
        backlog term Xqt of the routing weight; charging it again here
        would double-count the same shortage.
        """
        if demand <= 0 or not self.units:
            return 0.0
        left, wsum, tot = demand, 0.0, 0.0
        for g, a, _ in reversed(self.units):     # freshest first
            use = min(a, left)
            wsum += (t - g) * use
            tot += use
            left -= use
            if left <= 1e-12:
                break
        return wsum / tot if tot > 0 else 0.0

    def consume(self, t: int, amount: float) -> tuple[float, float, float]:
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
class HopQueue:
    """Blocks waiting at one arc, held in extended-nearest-to-origin order.

    ENTO priority is the number of arcs a block has already crossed, and
    that number is small and bounded by the longest admissible route, so
    the queue is a bucket per hop rather than a sorted list.  Insertion
    and service are O(1) amortized.  Sorting instead is correct but
    quadratic over a run, because an overloaded arc accumulates backlog
    linearly in time and would be resorted every slot.

    Within a bucket the order is first in, first out, which is what a
    stable sort on the hop count would also give.
    """

    __slots__ = ("buckets", "amount")

    def __init__(self) -> None:
        self.buckets: dict[int, deque] = {}
        self.amount = 0.0          # running total, so weights stay O(1)

    def __len__(self) -> int:
        return sum(len(b) for b in self.buckets.values())

    def __iter__(self):
        for h in sorted(self.buckets):
            yield from self.buckets[h]

    def append(self, b) -> None:
        self.buckets.setdefault(b.hop, deque()).append(b)
        self.amount += b.amount

    def head(self):
        """The block ENTO would serve next, or None."""
        for h in sorted(self.buckets):
            q = self.buckets[h]
            if q:
                return q[0]
            del self.buckets[h]
            return self.head()
        return None

    def take(self, b, got: float) -> None:
        """Record that `got` of block `b` at the head was served."""
        self.amount -= got
        if got >= b.amount - 1e-12:
            self.buckets[b.hop].popleft()
            if not self.buckets[b.hop]:
                del self.buckets[b.hop]
        else:
            b.amount -= got


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
        self.qkd_edges_of = qkd_edges_by_station(edges)
        # Transformed arcs that can never be served are deleted, not
        # merely disfavoured.  An edge outside the quantum overlay has
        # eta_e = 0, so its q-arc has service rate identically zero and a
        # graded block routed onto it waits forever.  Leaving the arc in
        # place lets Dijkstra pick it whenever the queue terms make it
        # look cheap, which strands traffic on a route that no feasible
        # decomposition would ever use.
        self.q_arcs = {e.key() for e in edges if e.qkd_capable}

        ek = [e.key() for e in edges]
        self.C = {k: bps_to_blocks(self.eidx[k].capacity_bps, cfg.dt_s)
                  for k in ek}
        self.Xq = {k: HopQueue() for k in ek}
        self.Xp = {k: HopQueue() for k in ek}
        self.Y = {k: HopQueue() for k in ek}
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
            # the age of the material this edge is about to spend, not
            # of the material it happens to hold
            demand = max(self.Xq[k].amount, 1.0)
            age = self.bq[k].marginal_age(t, demand)
            cq = self.Xqt[k] + cfg.V_chi * min(age, cfg.T_a)
            cp = self.Xpt[k] + cfg.V_chi * cfg.T_a
            has_q = k in self.q_arcs
            if has_q:
                Wq[k] = self.Yt[k] + cq            # graded traffic only
            if cp < cq or not has_q:
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
    def manufacture_weights(self) -> dict:
        """The weights of the manufacturing subproblem, w_e = Xpt_e - V.nu.c_e."""
        return {e.key(): self.Xpt[e.key()]
                - self.cfg.V_nu * OPS_PER_KEY_UNIT for e in self.edges}

    def _manufacture_greedy(self) -> dict:
        """The per-node ratio-greedy fill, which is exact only one-sided.

        Kept so that the cost of using it anyway can be measured.  See
        `manufacture_gap`: once both endpoints are charged it is exact
        at nominal load, where no node budget binds two competing edges
        at once, and forfeits up to half the max-weight objective under
        stress, where they do.
        """
        cfg = self.cfg
        h = {e.key(): 0.0 for e in self.edges}
        if not cfg.manufacture:
            return h
        w = self.manufacture_weights()
        left = {n: cfg.node_budget_units for n in self.nodes}
        for k in sorted(w, key=lambda x: -w[x] / OPS_PER_KEY_UNIT):
            if w[k] <= 0:
                break
            cap = min(cfg.edge_h_max_units,
                      left[k[0]] / OPS_TAIL if OPS_TAIL > 0 else math.inf,
                      left[k[1]] / OPS_HEAD if OPS_HEAD > 0 else math.inf)
            if cap <= 1e-12:
                continue
            h[k] = cap
            left[k[0]] -= OPS_TAIL * cap
            left[k[1]] -= OPS_HEAD * cap
        return h

    def _manufacture(self) -> dict:
        """Maximise sum_e w_e h_e over the budget polytope H, exactly.

        Charging the KEM at both endpoints costs the per-node
        decomposition: H is no longer a Cartesian product over nodes, so
        the closed-form knapsack of the one-sided model is a heuristic
        here and not a good one.  The drift argument needs the maximum,
        not an approximation of it, so this solves the packing program
        of Proposition 2(iii).

        What survives from the closed form is the pruning rule.  Every
        edge whose waiting backlog fails to cover the marginal cost is
        set to zero, because raising it cannot raise the objective and
        only consumes budget at two nodes, and that is demand gating.
        """
        if not self.cfg.manufacture:
            return {e.key(): 0.0 for e in self.edges}
        h, _ = manufacture_lp(self.manufacture_weights(), self.cfg,
                              self.nodes, [e.key() for e in self.edges])
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
        qkd_units = defaultdict(float)
        for gs, inc in self.qkd_edges_of.items():
            rate = qkd_rate_at(t, self.qkd_schedule.get(gs, [])) * wmult
            if rate <= 0 or not inc:
                continue
            # One ground station runs one QKD terminal, so a pass yields
            # ONE key stream that has to be divided among the station's
            # quantum-capable edges.  Crediting each incident edge with
            # the full pass yield, as earlier versions did, manufactures
            # entropy out of node degree and inflated eta_e fourfold.
            share = keybps_to_units(rate, cfg.dt_s) / len(inc)
            for k in inc:
                qkd_units[k] += share
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
                # Extended nearest-to-origin, on the internal arcs too.
                # The prior work could serve these in any order because
                # it stored keys indefinitely and could then dominate the
                # physical queue by the virtual one pathwise.  Expiring
                # keys breaks that domination, so physical rate stability
                # has to come from the ENTO sample-path argument instead,
                # and that argument needs the discipline here as well.
                while budget > 1e-9:
                    b = queue.head()
                    if b is None:
                        break
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
                    queue.take(b, got)

        for k in self.C:
            self.cum_servedQ[k] += dq[k]
            self.cum_servedP[k] += dp[k]

        # 5. forward --------------------------------------------------------
        for k in self.C:
            budget = self.C[k]
            q = self.Y[k]
            while budget > 1e-9:                 # nearest to origin first
                b = q.head()
                if b is None:
                    break
                use = min(b.amount, budget)
                budget -= use
                q.take(b, use)
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
                xq=self.Xq[k].amount, xp=self.Xp[k].amount,
                y=self.Y[k].amount,
                servedQ=dq[k], servedP=dp[k],
                # the surrogate the routing weight actually reads
                ageQ=self.bq[k].marginal_age(t, max(self.Xq[k].amount, 1.0)),
                ageMean=self.bq[k].mean_age(t),
                ageOldest=self.bq[k].oldest_age(t)))

        phys = sum(self.Xq[k].amount + self.Xp[k].amount + self.Y[k].amount
                   for k in self.C)
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
    inc_of = qkd_edges_by_station(edges)
    tot = defaultdict(float)
    for t in range(horizon):
        for g, inc in inc_of.items():
            r = qkd_rate_at(t, schedule.get(g, []))
            if r <= 0 or not inc:
                continue
            share = keybps_to_units(r, dt) / len(inc)   # one pass, one stream
            for k in inc:
                tot[k] += share
    return {e.key(): tot[e.key()] / horizon for e in edges}


def manufacture_lp(w: dict, cfg: TqdConfig, nodes, eks) -> tuple[dict, float]:
    """Exact optimum of max_{h in H} sum_e w_e h_e, both endpoints charged.

    H is the packing polytope
        sum_{e in O(i)} OPS_TAIL h_e + sum_{e in I(i)} OPS_HEAD h_e <= B_i,
        0 <= h_e <= h_max_e.
    Each column has exactly two nonzero constraint coefficients, one per
    endpoint, so this is a fractional packing problem on the node-edge
    incidence matrix rather than a set of independent knapsacks.
    """
    from scipy.optimize import linprog
    if not cfg.manufacture:
        return {k: 0.0 for k in eks}, 0.0
    # Proposition 2(i): an edge with a nonpositive weight is zero at
    # every optimum, so drop it before solving.  Under demand gating
    # this removes most of the columns.
    live = [k for k in eks if w[k] > 0.0]
    out = {k: 0.0 for k in eks}
    if not live:
        return out, 0.0
    inc = _incidence(nodes, live)
    hi = cfg.edge_h_max_units
    res = linprog(np.array([-w[k] for k in live]),
                  A_ub=inc, b_ub=np.full(inc.shape[0], cfg.node_budget_units),
                  bounds=[(0.0, hi)] * len(live), method="highs")
    if not res.success:
        return out, 0.0
    for i, k in enumerate(live):
        out[k] = float(res.x[i])
    return out, float(-res.fun)


_INC_CACHE: dict = {}


def _incidence(nodes, eks) -> np.ndarray:
    """Node-by-edge KEM-cost matrix, both endpoints charged."""
    key = (tuple(nodes), tuple(eks))
    hit = _INC_CACHE.get(key)
    if hit is not None:
        return hit
    ni = {n: i for i, n in enumerate(nodes)}
    m = np.zeros((len(nodes), len(eks)))
    for j, k in enumerate(eks):
        m[ni[k[0]], j] += OPS_TAIL
        m[ni[k[1]], j] += OPS_HEAD
    if len(_INC_CACHE) > 4096:
        _INC_CACHE.clear()
    _INC_CACHE[key] = m
    return m


def manufacture_gap(net) -> tuple[float, float]:
    """Relative shortfall of the per-node greedy against the exact LP.

    Reported rather than assumed: the greedy is exact only when the
    budget constraints decouple across nodes, which charging both
    endpoints of the key exchange breaks.  Returns (worst, mean).
    """
    eks = [e.key() for e in net.edges]
    w = net.manufacture_weights()
    _, best = manufacture_lp(w, net.cfg, net.nodes, eks)
    if best <= 1e-9:
        return 0.0, 0.0
    got = sum(w[k] * v for k, v in net._manufacture_greedy().items())
    g = (best - got) / best
    return g, g


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
    for n in nodes:                                # per-node KEM budget
        row = np.zeros(nv)                         # both endpoints charged
        for k in eks:
            if k[0] == n:
                row[len(cols) + eidx[k]] += OPS_TAIL
            if k[1] == n:
                row[len(cols) + eidx[k]] += OPS_HEAD
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
