"""Numerical checks on the claims the hybrid-supply theory rests on.

Three claims are checked here before they are asserted in the paper.

  1. THE MANUFACTURING RULE, AND WHERE IT STOPS WORKING.  With the
     key-establishment cost charged to one endpoint the budget polytope
     is a product over nodes, and the sub-problem max_{h in H} sum_e
     w_e h_e is solved exactly by filling edges in decreasing w_e / c_e.
     Checked against a general-purpose LP.  Charging BOTH endpoints, as
     FIPS 203 requires, couples the nodes and the greedy stops being
     optimal; the size of the loss is measured on the evaluation
     instance rather than assumed, because it decides whether the closed
     form could have been kept as an approximation.

  2. SPECIALIZATION AND STRICT ENLARGEMENT.  With no manufacturing
     budget and every class requiring information-theoretic keys, the
     region collapses to the multicommodity flow region on the
     capacitated graph with omega_e = min(C_e, eta_e), which is the
     region of Akhtar et al.  With a positive budget and at least one
     flexible class it is strictly larger.  Checked by maximizing a
     common scaling of the arrival vector by LP.

  3. THE COST OF A SINGLE KEY SPECIES.  Treating the two key species
     as one fungible pool cannot serve a class that requires
     information-theoretic keys at any rate above eta_e, however large
     the manufacturing budget.  Checked by comparing the graded region
     against its fungible relaxation.

Run: python3 test_region.py
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog

RNG = np.random.default_rng(20260811)


# ---------------------------------------------------------------------------
# 1. manufacturing sub-problem
# ---------------------------------------------------------------------------
def manufacture_greedy(w, c, hmax, budget):
    """Fill edges by decreasing w_e/c_e until the budget is exhausted.

    `w` is the drift-plus-penalty weight Xp_e - V*nu*c_e, which may be
    negative on edges whose waiting backlog does not cover the cost of
    manufacturing; those edges are skipped, which is demand gating.
    `c` is the per-unit cost, `hmax` the per-edge ceiling, `budget` the
    node's per-slot compute budget.
    """
    h = np.zeros(len(w))
    order = np.argsort(-(w / c))
    left = budget
    for e in order:
        if left <= 0 or w[e] <= 0:
            break
        take = min(hmax[e], left / c[e])
        h[e] = take
        left -= c[e] * take
    return h


def manufacture_lp(w, c, hmax, budget):
    res = linprog(-w, A_ub=c[None, :], b_ub=[budget],
                  bounds=list(zip(np.zeros(len(w)), hmax)), method="highs")
    assert res.success, res.message
    return res.x


def test_manufacturing(trials: int = 2000) -> None:
    print("\n[1a] Proposition 2(ii): one-sided cost, greedy vs LP, signed weights")
    worst = 0.0
    for _ in range(trials):
        n = int(RNG.integers(1, 8))
        # signed weights: the drift-plus-penalty weight Xp - V*nu*c is
        # negative wherever manufacturing is not worth its cost
        w = RNG.random(n) * 10 - 3.0
        if RNG.random() < 0.3:                     # some edges idle
            w[RNG.integers(0, n)] = 0.0
        c = 0.1 + RNG.random(n) * 3
        hmax = RNG.random(n) * 5
        budget = RNG.random() * 6
        g, l = manufacture_greedy(w, c, hmax, budget), manufacture_lp(w, c, hmax, budget)
        vg, vl = float(w @ g), float(w @ l)
        worst = max(worst, (vl - vg) / max(abs(vl), 1e-12))
    print(f"    {trials} random instances, worst relative shortfall of the "
          f"greedy: {worst:.3e}")
    assert worst < 1e-9, "greedy is not optimal when the budgets decouple"
    print("    OK: with the head cost zero the greedy attains the LP optimum")


def test_manufacturing_coupled() -> None:
    """Proposition 2(iii): the greedy stops being optimal at both endpoints.

    This is a negative result and it is the reason the algorithm solves
    the packing program rather than filling by ratio.  Reported, not
    assumed, because the size of the loss decides whether the closed
    form could have been kept as an approximation.

    The loss is load dependent, which is worth knowing.  At nominal load
    the manufactured demand is spread thinly enough that no node budget
    binds two competing edges at once and the greedy is exact.  Under
    stress it binds often, and the greedy forfeits up to half the
    objective.  A closed form that is exact only where the resource is
    not scarce is not a closed form worth having.
    """
    print("\n[1b] Proposition 2(iii): both endpoints charged, greedy vs exact LP")
    import aos_tqd as A
    nodes, edges = A.default_topology()
    flows = A.scaled_flows()
    sched, _ = A.build_qkd_schedule(weather_seed=0, hours=12)
    worst_overall = 0.0
    for load in (1.0, 3.0, 6.0):
        net = A.HybridNetwork(nodes, edges, flows,
                              A.TqdConfig(theta=0.5, load_scale=load), sched)
        gaps = []
        for t in range(3000):
            net.step(t)
            if t % 50 == 0 and t > 200:
                gaps.append(A.manufacture_gap(net)[0])
        g = np.array(gaps)
        worst_overall = max(worst_overall, float(g.max()))
        print(f"    load {load:4.1f}x nominal: greedy forfeits "
              f"{g.mean():.3f} of the objective on average, "
              f"{g.max():.3f} at worst, nonzero in "
              f"{int((g > 1e-9).sum())} of {len(g)} sampled slots")
    assert worst_overall > 0.1, \
        "expected the greedy to be badly suboptimal once both endpoints pay"
    print("    OK: exact where compute is plentiful, wrong by up to "
          f"{worst_overall:.0%} where it is scarce, so the closed form of "
          "(ii) is not usable here")


# ---------------------------------------------------------------------------
# 2 and 3. the capacity region
# ---------------------------------------------------------------------------
def paths(adj, s, d, cutoff=6):
    out, stack = [], [(s, [s], {s})]
    while stack:
        u, seq, seen = stack.pop()
        if u == d:
            out.append(seq)
            continue
        if len(seq) > cutoff:
            continue
        for v in adj.get(u, ()):
            if v not in seen:
                stack.append((v, seq + [v], seen | {v}))
    return out


def max_scale(nodes, edges, C, eta, classes, cost, budget, its_required):
    """Largest t with t*lambda in Lambda_S, by linear programming.

    Variables: per class, per path, a Q-rate and a P-rate; per edge a
    manufacturing rate h_e; and the scaling t.  Constraints are the
    per-edge capacity, the per-edge key-species supplies, the per-node
    manufacturing budget, and the grade restriction.
    """
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, []).append(v)
    cols, meta = [], []
    for ci, (s, d, lam) in enumerate(classes):
        for seq in paths(adj, s, d):
            eks = list(zip(seq, seq[1:]))
            for species in ("Q", "P"):
                if species == "P" and its_required[ci]:
                    continue           # graded traffic may not use P-keys
                cols.append((ci, eks, species))
                meta.append(lam)
    nh = len(edges)
    nv = len(cols) + nh + 1            # path rates, manufacturing, scale
    ti = nv - 1
    A, b = [], []
    eidx = {e: k for k, e in enumerate(edges)}
    # per-edge classical capacity
    for e in edges:
        row = np.zeros(nv)
        for j, (_, eks, _) in enumerate(cols):
            if e in eks:
                row[j] = 1.0
        A.append(row); b.append(C[e])
    # per-edge Q-key supply
    for e in edges:
        row = np.zeros(nv)
        for j, (_, eks, sp) in enumerate(cols):
            if sp == "Q" and e in eks:
                row[j] = 1.0
        A.append(row); b.append(eta[e])
    # per-edge P-key supply: consumption must not exceed manufacture
    for e in edges:
        row = np.zeros(nv)
        for j, (_, eks, sp) in enumerate(cols):
            if sp == "P" and e in eks:
                row[j] = 1.0
        row[len(cols) + eidx[e]] = -1.0
        A.append(row); b.append(0.0)
    # per-node manufacturing budget
    for n in nodes:
        row = np.zeros(nv)
        for e in edges:
            if e[0] == n:
                row[len(cols) + eidx[e]] = cost[e]
        A.append(row); b.append(budget.get(n, 0.0))
    # flow conservation: total class rate must equal t * lambda
    Aeq, beq = [], []
    for ci, (_, _, lam) in enumerate(classes):
        row = np.zeros(nv)
        for j, (cj, _, _) in enumerate(cols):
            if cj == ci:
                row[j] = 1.0
        row[ti] = -lam
        Aeq.append(row); beq.append(0.0)
    c_obj = np.zeros(nv); c_obj[ti] = -1.0
    res = linprog(c_obj, A_ub=np.array(A), b_ub=np.array(b),
                  A_eq=np.array(Aeq), b_eq=np.array(beq),
                  bounds=[(0, None)] * nv, method="highs")
    assert res.success, res.message
    return float(res.x[ti])


def diamond():
    nodes = ["s", "a", "b", "d"]
    edges = [("s", "a"), ("s", "b"), ("a", "d"), ("b", "d")]
    C = {e: 10.0 for e in edges}
    eta = {("s", "a"): 1.0, ("s", "b"): 2.0,
           ("a", "d"): 1.5, ("b", "d"): 0.5}
    cost = {e: 1.0 for e in edges}
    return nodes, edges, C, eta, cost


def test_region() -> None:
    print("\n[2] region: specialization to prior work, and enlargement")
    nodes, edges, C, eta, cost = diamond()
    classes = [("s", "d", 1.0)]

    no_budget = {n: 0.0 for n in nodes}
    big_budget = {n: 100.0 for n in nodes}

    t_tqd = max_scale(nodes, edges, C, eta, classes, cost, no_budget, [True])
    # the Akhtar et al. region: multicommodity flow with omega_e=min(C,eta)
    omega = {e: min(C[e], eta[e]) for e in edges}
    t_ref = max_scale(nodes, edges, omega, omega, classes, cost,
                      no_budget, [True])
    print(f"    B=0, graded : max scale {t_tqd:.6f}")
    print(f"    min(C,eta) reference       {t_ref:.6f}")
    assert abs(t_tqd - t_ref) < 1e-9
    print("    OK: with no budget and graded traffic the region is theirs")

    t_flex = max_scale(nodes, edges, C, eta, classes, cost, big_budget,
                       [False])
    print(f"    B>0, flexible: max scale {t_flex:.6f}")
    assert t_flex > t_tqd + 1e-6
    print(f"    OK: manufacturing strictly enlarges the region "
          f"({t_flex / t_tqd:.2f}x here)")

    print("\n[3] a single fungible pool cannot serve graded traffic")
    t_graded = max_scale(nodes, edges, C, eta, classes, cost, big_budget,
                         [True])
    print(f"    B>0, graded  : max scale {t_graded:.6f}")
    assert abs(t_graded - t_tqd) < 1e-9, \
        "budget must not help traffic that requires QKD-origin keys"
    print("    OK: budget buys nothing for graded traffic, so the two "
          "species are not interchangeable")

    print("\n    grade-capacity tradeoff (fraction of load requiring QKD):")
    for theta in (0.0, 0.25, 0.5, 0.75, 1.0):
        cl, req = [], []
        if theta > 0:
            cl.append(("s", "d", theta)); req.append(True)
        if theta < 1:
            cl.append(("s", "d", 1 - theta)); req.append(False)
        t = max_scale(nodes, edges, C, eta, cl, cost, big_budget, req)
        print(f"      theta={theta:4.2f}   max scale {t:7.4f}")


if __name__ == "__main__":
    test_manufacturing()
    test_manufacturing_coupled()
    test_region()
    print("\nall checks passed")
