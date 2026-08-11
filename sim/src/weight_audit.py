"""Audit of the routing weight's three terms under AoS-BP.

Three questions the manuscript answers numerically, all of them about
the edge cost psi_e = omega*rho*Z_e + beta'*L_e + chi'*AoS_e:

  1. How is psi_e divided among its terms, on the edge-slots where the
     key-deficit term is active at all?
  2. What does omega actually control?  Sweeping it over five decades
     separates "inert parameter" from "the knob that steers traffic off
     a starved edge".
  3. How often is the positivity condition W_psi > 0 binding?  Under
     AoS-BP it is not imposed but implied (Theorem 3(iv)), so the
     question is how often the linear program declines to serve a flow
     that has backlog.

Run: python3 weight_audit.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import aos_network as A

OUT = Path("/tmp/aos_weight_audit")
RHO = A.KEY_BITS_PER_PAYLOAD_BIT


def instrument(cfg: A.SimConfig, sched: dict) -> dict:
    nodes, edges = A.default_topology()
    net = A.AoSNetwork(nodes, edges, A.default_flows(), cfg, sched)
    parts, gate = [], []
    orig = net._cg_plan

    def hook(rbar, psi, qsrc):
        for ek in rbar:
            e = net.edge_index[ek]
            z = cfg.bp_omega * RHO * net.Z[ek]
            b = cfg.bp_beta_prime * e.latency_ms
            c = cfg.bp_chi_prime * net.edge_aos(e, cfg._t)
            parts.append((z, b, c))
        for f in net.flows:
            q = qsrc.get(f.name, 0.0)
            if q <= 0:
                continue
            nodes_p = net._dijkstra_psi(f, set(rbar), psi)
            if len(nodes_p) < 2:
                gate.append(np.nan)
                continue
            eks = list(zip(nodes_p, nodes_p[1:]))
            gate.append(q - sum(psi[ek] for ek in eks))
        return orig(rbar, psi, qsrc)

    net._cg_plan = hook
    for t in range(cfg.horizon_s):
        cfg._t = t
        net.step(t)
    p = np.array(parts, dtype=float)
    tot = p.sum(axis=1)
    live = p[:, 0] > 0
    g = np.array([x for x in gate if not np.isnan(x)], dtype=float)
    return dict(
        aos=float(np.mean([l.aos_mean for l in net.logs])),
        goodput=sum(l.total_secure_bits for l in net.logs) / cfg.horizon_s,
        frac_live=float(live.mean()),
        share_z=float((p[live, 0] / tot[live]).mean()) if live.any() else 0.0,
        share_b=float((p[live, 1] / tot[live]).mean()) if live.any() else 0.0,
        share_c=float((p[live, 2] / tot[live]).mean()) if live.any() else 0.0,
        gate_block_frac=float((g <= 0).mean()) if len(g) else 0.0,
        n_flow_slots=len(g),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [0, 1, 2]
    sched = {s: A.build_qkd_schedule(weather_seed=s, hours=12)[0]
             for s in seeds}

    print("psi decomposition and gate activity, AoS-BP at the operating "
          "point")
    rows = [instrument(A.SimConfig(horizon_s=600, seed=s, scheduler="aos_cg",
                                   scenario="nominal"), sched[s])
            for s in seeds]
    m = lambda k: float(np.mean([r[k] for r in rows]))
    print(f"  edge-slots with a key deficit : {100*m('frac_live'):.1f}%")
    print(f"  on those, share of psi from omega*rho*Z : "
          f"{100*m('share_z'):.1f}%")
    print(f"                       from beta'*L      : "
          f"{100*m('share_b'):.1f}%")
    print(f"                       from chi'*AoS     : "
          f"{100*m('share_c'):.1f}%")
    print(f"  flow-slots with W_psi <= 0 on the min-psi path : "
          f"{100*m('gate_block_frac'):.2f}%")

    print("\nomega sweep, AoS-BP, nominal")
    print(f"  {'omega':>10s} {'meanAoS':>9s} {'goodput':>10s}")
    for om in (2e0, 2e1, 2e2, 2e3, 2e4, 2e5):
        rs = [instrument(A.SimConfig(horizon_s=600, seed=s,
                                     scheduler="aos_cg", scenario="nominal",
                                     bp_omega=om), sched[s]) for s in seeds]
        print(f"  {om:10.0e} {np.mean([r['aos'] for r in rs]):9.2f} "
              f"{np.mean([r['goodput'] for r in rs])/1e6:10.2f}")


if __name__ == "__main__":
    main()
