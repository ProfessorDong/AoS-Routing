"""How much traffic a pooled-key policy certifies wrongly.

A policy that keeps one key bank per link and treats quantum and
manufactured material as interchangeable will report every block it
serves from that bank as quantum-keyed.  With per-unit provenance in the
banks the claim can be checked against what was actually spent.  This
sweeps offered load and graded fraction, since the error should grow
wherever quantum supply is scarce relative to demand.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import aos_tqd as A

OUT = Path(__file__).resolve().parent.parent / "results_tqd"
SEEDS, HORIZON = [0, 1, 2], 43200

sched = {s: A.build_qkd_schedule(weather_seed=s, hours=12)[0] for s in SEEDS}
rows = []
print(f"{'load':>5s} {'theta':>5s} {'policy':>9s} {'goodput':>8s} "
      f"{'trueITS':>8s} {'claimed':>8s} {'mislbl':>7s}", flush=True)
for ls in (1.0, 2.0, 3.0):
    for th in (0.25, 0.5, 0.75):
        for pol in ("aos_tqd", "fungible"):
            r = [A.run(A.TqdConfig(horizon_s=HORIZON, seed=s, theta=th,
                                   policy=pol, load_scale=ls), sched[s])
                 for s in SEEDS]
            m = lambda k: float(np.mean([x[k] for x in r]))
            rows.append(dict(load_scale=ls, theta=th, policy=pol,
                             goodput_mbps=m("goodput_mbps"),
                             its_fraction=m("its_fraction"),
                             claimed_its_fraction=m("claimed_its_fraction"),
                             mislabelled_fraction=m("mislabelled_fraction")))
            print(f"{ls:5.1f} {th:5.2f} {pol:>9s} {m('goodput_mbps'):8.2f} "
                  f"{m('its_fraction'):8.3f} {m('claimed_its_fraction'):8.3f} "
                  f"{m('mislabelled_fraction'):7.3f}", flush=True)
with open(OUT / "mislabel.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("wrote mislabel.csv")
