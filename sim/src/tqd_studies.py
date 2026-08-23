"""Parameter and boundary studies for the hybrid-supply policy.

Five studies.  Each one is paired with the linear program of
`region_boundary`, so the measured behaviour is compared against the
region the theory predicts rather than against a fitted curve.

  A  LOAD SWEEP.  The point of manufacturing is that it enlarges the
     capacity region, and that is invisible at an operating point inside
     both regions.  Sweeping the offered load past the quantum-only
     boundary is what separates the policies.

  B  BUDGET SWEEP.  How the region and the delivered goodput grow with
     the per-node encapsulation budget, at a load chosen to lie outside
     the quantum-only region.

  C  GRADE SWEEP.  The measured counterpart of the reciprocal in
     Theorem 3.

  D  FRESHNESS WEIGHT.  Selection of V*chi by the same rule used for
     the conference algorithm: the knee at which mean Age of Secret
     stops falling while backlog is still flat.

  E  MANUFACTURING COST.  Sweeping V*nu traces demand gating, which the
     drift-plus-penalty derivation produces rather than assumes: edges
     whose waiting backlog does not cover the marginal cost of an
     encapsulation are skipped.

Run: python3 tqd_studies.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import aos_tqd as A

OUT = Path(__file__).resolve().parent.parent / "results_tqd"
SEEDS = [0, 1, 2]
# The whole pass schedule, so that the time average of the supply over a
# run IS the stationary mean the capacity region is written in.  Over a
# shorter window the two differ by up to 40 percent here, because
# satellite supply is strongly non-stationary within a day, and the
# measured boundary would then be compared against the wrong number.
HORIZON = 43200


def _mean(rows, k):
    return float(np.mean([r[k] for r in rows]))


def _run(sched, **kw):
    return [A.run(A.TqdConfig(horizon_s=HORIZON, seed=s, **kw), sched[s])
            for s in SEEDS]


def _boundary(eta, edges, nodes, flows, **kw):
    return A.region_boundary(eta, A.TqdConfig(**kw), flows, edges, nodes)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sched = {s: A.build_qkd_schedule(weather_seed=s, hours=12)[0]
             for s in SEEDS}
    nodes, edges = A.default_topology()
    flows = A.scaled_flows()
    # eta averaged over exactly the simulated window and over seeds, so
    # the linear program and the runs refer to the same supply process
    etas = [A.empirical_eta(sched[s], HORIZON, edges) for s in SEEDS]
    eta = {k: float(np.mean([e[k] for e in etas])) for k in etas[0]}
    nominal = sum(f.arrival_bps for f in flows) / 1e6

    fields = ["study", "policy", "theta", "load_scale", "budget_scale",
              "V_chi", "V_nu", "offered_mbps", "lp_boundary_mbps",
              "goodput_mbps", "its_fraction", "mislabelled_fraction",
              "mean_aos", "mean_delay", "phys_slope",
              "manufacture_utilisation", "qkd_expired_frac"]
    rows = []

    def record(study, r, **extra):
        row = dict(study=study, policy=r["policy"], theta=r["theta"],
                   load_scale=r["load_scale"],
                   goodput_mbps=r["goodput_mbps"],
                   its_fraction=r["its_fraction"],
                   mislabelled_fraction=r["mislabelled_fraction"],
                   mean_aos=r["mean_aos"], mean_delay=r["mean_delay"],
                   phys_slope=r["phys_slope"],
                   manufacture_utilisation=r["manufacture_utilisation"],
                   qkd_expired_frac=r["qkd_expired_frac"])
        row.update(extra)          # seed-averaged values override seed 0
        rows.append(row)

    # ---- A. load sweep -------------------------------------------------
    print("A load sweep (theta = 1/2)")
    for ls in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        for pol in ("aos_tqd", "tqd"):
            th = 0.5 if pol == "aos_tqd" else 1.0
            mf = pol == "aos_tqd"
            b = _boundary(eta, edges, nodes, flows, theta=th,
                          manufacture=mf) * nominal
            r = _run(sched, policy=pol, theta=th, load_scale=ls,
                     scenario="nominal")
            record("A", r[0], V_chi=A.TqdConfig().V_chi, V_nu=0.0,
                   budget_scale=1.0 if mf else 0.0,
                   offered_mbps=nominal * ls, lp_boundary_mbps=b,
                   goodput_mbps=_mean(r, "goodput_mbps"),
                   phys_slope=_mean(r, "phys_slope"))
            print(f"  {pol:>8s} load {nominal*ls:6.2f} Mbps  "
                  f"boundary {b:6.2f}  goodput "
                  f"{_mean(r,'goodput_mbps'):6.2f}  slope "
                  f"{_mean(r,'phys_slope'):10.2e}")

    # ---- B. budget sweep ------------------------------------------------
    print("\nB budget sweep (theta = 1/2, load 2x nominal)")
    base = A.TqdConfig().node_budget_units
    for bs in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        b = _boundary(eta, edges, nodes, flows, theta=0.5,
                      manufacture=bs > 0,
                      node_budget_units=base * bs,
                      edge_h_max_units=base * bs) * nominal
        r = _run(sched, policy="aos_tqd", theta=0.5, load_scale=2.0,
                 scenario="nominal", node_budget_units=base * bs,
                 edge_h_max_units=base * bs)
        record("B", r[0], V_chi=A.TqdConfig().V_chi, V_nu=0.0,
               budget_scale=bs, offered_mbps=nominal * 2.0,
               lp_boundary_mbps=b,
               goodput_mbps=_mean(r, "goodput_mbps"),
               phys_slope=_mean(r, "phys_slope"),
               its_fraction=_mean(r, "its_fraction"))
        print(f"  budget {bs:4.2f}x  boundary {b:6.2f}  goodput "
              f"{_mean(r,'goodput_mbps'):6.2f}  ITS "
              f"{_mean(r,'its_fraction'):5.3f}")

    # ---- C. grade sweep -------------------------------------------------
    print("\nC grade sweep (load 2x nominal)")
    for th in (0.0, 0.125, 0.25, 0.5, 0.75, 1.0):
        b = _boundary(eta, edges, nodes, flows, theta=th) * nominal
        r = _run(sched, policy="aos_tqd", theta=th, load_scale=2.0,
                 scenario="nominal")
        record("C", r[0], V_chi=A.TqdConfig().V_chi, V_nu=0.0,
               budget_scale=1.0, offered_mbps=nominal * 2.0,
               lp_boundary_mbps=b,
               goodput_mbps=_mean(r, "goodput_mbps"),
               its_fraction=_mean(r, "its_fraction"),
               mean_aos=_mean(r, "mean_aos"),
               phys_slope=_mean(r, "phys_slope"))
        print(f"  theta {th:5.3f}  boundary {b:6.2f}  goodput "
              f"{_mean(r,'goodput_mbps'):6.2f}  AoS "
              f"{_mean(r,'mean_aos'):7.2f}")

    # ---- D. freshness weight --------------------------------------------
    print("\nD freshness weight V.chi (theta = 1/2, nominal load)")
    for vc in (0.0, 0.1, 1.0, 10.0, 100.0, 1000.0):
        r = _run(sched, policy="aos_tqd", theta=0.5, load_scale=1.0,
                 scenario="nominal", V_chi=vc)
        record("D", r[0], V_chi=vc, V_nu=0.0, budget_scale=1.0,
               offered_mbps=nominal, lp_boundary_mbps=float("nan"),
               goodput_mbps=_mean(r, "goodput_mbps"),
               mean_aos=_mean(r, "mean_aos"),
               mean_delay=_mean(r, "mean_delay"),
               its_fraction=_mean(r, "its_fraction"),
               phys_slope=_mean(r, "phys_slope"))
        print(f"  V.chi {vc:8.1f}  AoS {_mean(r,'mean_aos'):7.2f}  "
              f"goodput {_mean(r,'goodput_mbps'):6.2f}  ITS "
              f"{_mean(r,'its_fraction'):5.3f}  slope "
              f"{_mean(r,'phys_slope'):10.2e}")

    # ---- E. manufacturing cost, that is demand gating --------------------
    print("\nE manufacturing cost V.nu (theta = 1/2, load 2x nominal)")
    for vn in (0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0):
        r = _run(sched, policy="aos_tqd", theta=0.5, load_scale=2.0,
                 scenario="nominal", V_nu=vn)
        record("E", r[0], V_chi=A.TqdConfig().V_chi, V_nu=vn,
               budget_scale=1.0, offered_mbps=nominal * 2.0,
               lp_boundary_mbps=float("nan"),
               goodput_mbps=_mean(r, "goodput_mbps"),
               manufacture_utilisation=_mean(r, "manufacture_utilisation"),
               its_fraction=_mean(r, "its_fraction"),
               phys_slope=_mean(r, "phys_slope"))
        print(f"  V.nu {vn:9.1f}  manufactured "
              f"{_mean(r,'manufacture_utilisation'):6.4f}  goodput "
              f"{_mean(r,'goodput_mbps'):6.2f}  ITS "
              f"{_mean(r,'its_fraction'):5.3f}")

    with open(OUT / "studies.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT / 'studies.csv'}")


if __name__ == "__main__":
    main()
