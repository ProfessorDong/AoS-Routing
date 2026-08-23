"""Extend the load sweep past this paper's own boundary.

The main sweep stops at three times nominal, which brackets the
quantum-only boundary but not ours, so the figure would show only one
of the two liftoffs.  These points close it.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import aos_tqd as A

OUT = Path(__file__).resolve().parent.parent / "results_tqd"
SEEDS, HORIZON = [0, 1, 2], 43200

sched = {s: A.build_qkd_schedule(weather_seed=s, hours=12)[0] for s in SEEDS}
nodes, edges = A.default_topology()
flows = A.scaled_flows()
etas = [A.empirical_eta(sched[s], HORIZON, edges) for s in SEEDS]
eta = {k: float(np.mean([e[k] for e in etas])) for k in etas[0]}
nominal = sum(f.arrival_bps for f in flows) / 1e6

rows = []
for ls in (3.5, 4.0, 4.5, 5.0):
    for pol in ("aos_tqd", "tqd"):
        th, mf = (0.5, True) if pol == "aos_tqd" else (1.0, False)
        b = A.region_boundary(eta, A.TqdConfig(theta=th, manufacture=mf),
                              flows, edges, nodes) * nominal
        r = [A.run(A.TqdConfig(horizon_s=HORIZON, seed=s, policy=pol,
                               theta=th, load_scale=ls,
                               scenario="nominal"), sched[s]) for s in SEEDS]
        m = lambda k: float(np.mean([x[k] for x in r]))
        rows.append(dict(study="A", policy=pol, theta=th, load_scale=ls,
                         budget_scale=1.0 if mf else 0.0,
                         V_chi=A.TqdConfig().V_chi, V_nu=0.0,
                         offered_mbps=nominal * ls, lp_boundary_mbps=b,
                         goodput_mbps=m("goodput_mbps"),
                         its_fraction=m("its_fraction"),
                         mislabelled_fraction=m("mislabelled_fraction"),
                         mean_aos=m("mean_aos"), mean_delay=m("mean_delay"),
                         phys_slope=m("phys_slope"),
                         manufacture_utilisation=m("manufacture_utilisation"),
                         qkd_expired_frac=m("qkd_expired_frac")))
        print(f"  {pol:>8s} load {nominal*ls:6.2f}  boundary {b:6.2f}  "
              f"goodput {m('goodput_mbps'):6.2f}  slope {m('phys_slope'):9.2e}",
              flush=True)

with open(OUT / "extra_loads.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("wrote extra_loads.csv")
