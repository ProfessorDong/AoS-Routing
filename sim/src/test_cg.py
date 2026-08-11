"""Correctness tests for the column-generation max-weight solver.

Three checks, in increasing strength:

  1. On the two-flow counterexample that breaks the per-flow greedy, the
     LP recovers the joint optimum and the greedy does not.
  2. On the real ten-node topology, at randomly drawn states, the LP
     objective matches a brute-force enumeration of all simple paths
     solved as a single dense LP.  This is the exactness certificate:
     column generation and full enumeration must agree to solver
     tolerance.
  3. The action the LP returns is feasible: no edge ceiling and no source
     backlog is exceeded.

Run: python3 test_cg.py
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.optimize import linprog

from aos_network import (KEY_BITS_PER_PAYLOAD_BIT, LP_SCALE, AoSNetwork,
                         Edge, Flow, SimConfig, default_flows,
                         default_topology)

RHO = KEY_BITS_PER_PAYLOAD_BIT


def make_net(nodes, edges, flows, cfg=None) -> AoSNetwork:
    cfg = cfg or SimConfig(scheduler="aos_cg")
    return AoSNetwork(nodes, edges, flows, cfg, {})


# ---------------------------------------------------------------------------
# 1. the counterexample
# ---------------------------------------------------------------------------
def test_counterexample() -> None:
    """Two flows, one shared key-limited edge, one private expensive edge.

    Flow 1 has the larger backlog, so the greedy serves it first and
    consumes the whole key budget on e0.  Flow 2, whose path is worth far
    more per unit rate, is left with nothing.  The gap grows with M.
    """
    print("\n[1] two-flow counterexample")
    M = 1.0e8
    nodes = ["s1", "s2", "a", "d"]
    # e0 is the contended edge; both flows must cross it.
    edges = [
        Edge("s1", "a", capacity_bps=1e12, latency_ms=0.0),
        Edge("s2", "a", capacity_bps=1e12, latency_ms=0.0),
        Edge("a", "d", capacity_bps=1e12, latency_ms=0.0),
    ]
    flows = [Flow("f1", "s1", "d", 0.0), Flow("f2", "s2", "d", 0.0)]
    cfg = SimConfig(scheduler="aos_cg", bp_beta_prime=0.0, bp_chi_prime=0.0,
                    bp_omega=1.0)
    net = make_net(nodes, edges, flows, cfg)
    # One unit of key on the shared edge, plenty elsewhere.
    unit = 1.0e6                       # payload bits the budget allows
    net.K[("a", "d")] = unit * RHO
    net.K[("s1", "a")] = 1e18
    net.K[("s2", "a")] = 1e18
    for ek in net.Z:
        net.Z[ek] = 0.0
    # f1 has the larger backlog; f2's path is worth much more per bit
    # because f1's private edge carries a large virtual-queue cost.
    net.Q[("f1", "s1")].append(M + 1.0)
    net.Q[("f2", "s2")].append(M)
    net.Z[("s1", "a")] = M / (cfg.bp_omega * RHO)   # omega*rho*Z = M

    rbar, psi, qsrc = net._slot_state(0, set())
    greedy = net._greedy_plan(rbar, psi, qsrc)
    plan, diag = net._cg_plan(rbar, psi, qsrc)
    g_obj = sum(r * (qsrc[f] - sum(psi[ek] for ek in eks))
                for f, eks, r in greedy)
    print(f"    greedy objective   {g_obj:.6e}")
    print(f"    LP objective       {diag['obj_lp']:.6e}")
    print(f"    ratio LP/greedy    {diag['obj_lp'] / max(g_obj, 1e-30):.1f}")
    assert diag["obj_lp"] > 0
    assert diag["obj_lp"] > 10 * max(g_obj, 1e-30), "LP should dominate here"
    print("    OK: greedy loses a factor that grows with M")


# ---------------------------------------------------------------------------
# 2. exactness against brute-force enumeration
# ---------------------------------------------------------------------------
def all_simple_paths(nodes, adj, src, dst, cutoff=None):
    """Every simple src->dst path.  `cutoff` bounds the node count; the
    default enumerates all of them, which is what the exactness test
    needs -- a shorter cutoff silently drops the long gateway-mesh
    detours that the LP is entitled to use."""
    cutoff = cutoff or len(nodes)
    out = []
    stack = [(src, [src], {src})]
    while stack:
        u, seq, seen = stack.pop()
        if u == dst:
            out.append(seq)
            continue
        if len(seq) > cutoff:
            continue
        for v in adj.get(u, ()):
            if v in seen:
                continue
            stack.append((v, seq + [v], seen | {v}))
    return out


def brute_force_optimum(net, rbar, psi, qsrc):
    """Solve the max-weight LP with EVERY simple path enumerated."""
    adj = {}
    for (u, v) in rbar:
        adj.setdefault(u, []).append(v)
    cols = []
    for f in net.flows:
        if qsrc[f.name] <= 0:
            continue
        for seq in all_simple_paths(net.nodes, adj, f.src, f.dst):
            eks = tuple(zip(seq, seq[1:]))
            if not eks or any(ek not in rbar for ek in eks):
                continue
            cols.append((f.name, eks,
                         qsrc[f.name] - sum(psi[ek] for ek in eks)))
    if not cols:
        return 0.0, 0
    edges_used = sorted({ek for _, eks, _ in cols for ek in eks})
    eidx = {ek: i for i, ek in enumerate(edges_used)}
    fnames = sorted({f for f, _, _ in cols})
    fidx = {fn: len(edges_used) + i for i, fn in enumerate(fnames)}
    A = np.zeros((len(edges_used) + len(fnames), len(cols)))
    b = np.empty(A.shape[0])
    for j, (fn, eks, _) in enumerate(cols):
        for ek in eks:
            A[eidx[ek], j] = 1.0
        A[fidx[fn], j] = 1.0
    for ek, i in eidx.items():
        b[i] = rbar[ek] / LP_SCALE
    for fn, i in fidx.items():
        b[i] = qsrc[fn] / LP_SCALE
    c = np.array([-w / LP_SCALE for _, _, w in cols])
    res = linprog(c, A_ub=A, b_ub=b, bounds=(0.0, None), method="highs")
    assert res.success, res.message
    return -float(res.fun) * LP_SCALE * LP_SCALE, len(cols)


def test_exactness(trials: int = 40) -> None:
    print("\n[2] column generation vs full path enumeration")
    nodes, edges = default_topology()
    flows = default_flows()
    rng = np.random.default_rng(20260811)
    worst = 0.0
    n_paths = 0
    for it in range(trials):
        net = make_net(nodes, edges, flows)
        for ek in net.K:
            net.K[ek] = float(10.0 ** rng.uniform(3.5, 8.5))
        for ek in net.Z:
            net.Z[ek] = float(10.0 ** rng.uniform(0.0, 7.0)
                              if rng.random() < 0.5 else 0.0)
        for f in flows:
            net.Q[(f.name, f.src)].append(float(10.0 ** rng.uniform(5, 9)))
        net.gen_epoch = {ek: -rng.uniform(0, 400) for ek in net.K}
        t = int(rng.integers(0, 600))
        block = set()
        if rng.random() < 0.3:
            block.add(rng.choice(["GW-EU", "GW-APAC", "GW-NAWest"]))
        rbar, psi, qsrc = net._slot_state(t, block)
        plan, diag = net._cg_plan(rbar, psi, qsrc)
        exact, ncols = brute_force_optimum(net, rbar, psi, qsrc)
        n_paths = max(n_paths, ncols)
        assert diag["converged"] == 1, "hit the iteration cap"
        denom = max(abs(exact), 1.0)
        rel = abs(diag["obj_lp"] - exact) / denom
        worst = max(worst, rel)
        # feasibility of the returned action
        load = {}
        served = {}
        for fn, eks, r in plan:
            served[fn] = served.get(fn, 0.0) + r
            for ek in eks:
                load[ek] = load.get(ek, 0.0) + r
        for ek, v in load.items():
            assert v <= rbar[ek] * (1 + 1e-7) + 1e-3, f"edge {ek} overloaded"
        for fn, v in served.items():
            assert v <= qsrc[fn] * (1 + 1e-7) + 1e-3, f"flow {fn} overserved"
    print(f"    {trials} random states, up to {n_paths} enumerated columns")
    print(f"    worst relative objective error: {worst:.3e}")
    assert worst < 1e-6, "column generation is not matching the exact LP"
    print("    OK: exact to solver tolerance, actions feasible")


# ---------------------------------------------------------------------------
# 3. the greedy gap on the real topology
# ---------------------------------------------------------------------------
def test_gap_on_topology(trials: int = 200) -> None:
    print("\n[3] greedy optimality gap at random states, real topology")
    nodes, edges = default_topology()
    flows = default_flows()
    rng = np.random.default_rng(7)
    gaps = []
    for _ in range(trials):
        net = make_net(nodes, edges, flows)
        for ek in net.K:
            net.K[ek] = float(10.0 ** rng.uniform(3.0, 8.0))
        for ek in net.Z:
            net.Z[ek] = float(10.0 ** rng.uniform(0.0, 7.0)
                              if rng.random() < 0.5 else 0.0)
        for f in flows:
            net.Q[(f.name, f.src)].append(float(10.0 ** rng.uniform(5, 9)))
        net.gen_epoch = {ek: -rng.uniform(0, 400) for ek in net.K}
        rbar, psi, qsrc = net._slot_state(int(rng.integers(0, 600)), set())
        _, diag = net._cg_plan(rbar, psi, qsrc)
        if diag["obj_lp"] > 0:
            gaps.append(1.0 - diag["obj_greedy"] / diag["obj_lp"])
    gaps = np.array(gaps)
    print(f"    n={len(gaps)}  mean {gaps.mean():.4f}  "
          f"median {np.median(gaps):.4f}  max {gaps.max():.4f}  "
          f"frac>1%: {(gaps > 0.01).mean():.3f}")


if __name__ == "__main__":
    test_counterexample()
    test_exactness()
    test_gap_on_topology()
    print("\nall checks passed")
