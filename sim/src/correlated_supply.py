"""What temporally correlated key supply does, measured against reanalysis.

Section V-F establishes that cloud optical depth over each of the six
sites has a lag-one autocorrelation between 0.88 and 0.93, so drawing an
independent attenuation for every satellite pass is the wrong model in
time.  This script replaces the draw with the reanalysis itself and asks
what changes.

Three things are separated here, because they are usually conflated.

  LEVEL.  Does correlated cloud deliver less key material on average?
  Averaged over the year it should not: correlation rearranges outages,
  it does not create them.

  DISPERSION.  Does the supply a given operating window actually sees
  match the long-run mean?  Under independent draws it does, because a
  12 h window contains thousands of independent passes.  Under real
  weather a window contains one or two independent weather states, so
  it need not.  This is the quantity the independent model destroys.

  CONSEQUENCE.  Setting the load from the long-run boundary and then
  running in a particular week is what an operator would do.  What does
  the resulting backlog look like, and is the policy still stable?

The orbital layer is held fixed at the element-set epoch throughout, so
that everything reported here is attributable to weather and not to
geometry.  Each realization reads a different four-week offset into the
reanalysis year.

Run:  python3 correlated_supply.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import aos_tqd as A

OUT = Path(__file__).resolve().parent.parent / "results_tqd"
HORIZON = 43200
# Thirteen four-week offsets step through the reanalysis year once.
REALIZATIONS = list(range(13))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nodes, edges = A.default_topology()
    flows = A.scaled_flows()
    nominal = sum(f.arrival_bps for f in flows) / 1e6

    etas, rows = {}, []
    for model in ("isccp", "era5"):
        per = []
        for s in REALIZATIONS:
            sched, _ = A.build_qkd_schedule(weather_seed=s, hours=12,
                                            weather_model=model)
            per.append(A.empirical_eta(sched, HORIZON, edges))
        etas[model] = per
        tot = [sum(e.values()) for e in per]
        print(f"{model:6s} network eta over {len(per)} realizations: "
              f"mean {np.mean(tot):8.1f}  cv {np.std(tot)/np.mean(tot):5.3f}  "
              f"min {min(tot):7.1f}  max {max(tot):8.1f} units/slot")

    # The long-run boundary uses the ensemble mean supply, which is the
    # stationary eta the region is written in.
    for model in ("isccp", "era5"):
        per = etas[model]
        eta_bar = {k: float(np.mean([e[k] for e in per])) for k in per[0]}
        tstar = A.region_boundary(eta_bar, A.TqdConfig(theta=0.5),
                                  flows, edges, nodes)
        # and the boundary each individual window would have supported
        win = [A.region_boundary(e, A.TqdConfig(theta=0.5), flows, edges,
                                 nodes) * nominal for e in per]
        print(f"\n{model}: long-run boundary {tstar*nominal:7.2f} Mbps; "
              f"per-window boundary mean {np.mean(win):7.2f} "
              f"min {min(win):6.2f} max {max(win):7.2f} Mbps")
        rows.append(dict(model=model, longrun_mbps=tstar * nominal,
                         window_mean_mbps=float(np.mean(win)),
                         window_min_mbps=float(min(win)),
                         window_max_mbps=float(max(win)),
                         window_cv=float(np.std(win) / max(np.mean(win),
                                                           1e-9))))

    # Consequence: hold the load at the ISCCP operating point and run
    # each weather realization of each model.
    load_mbps = A.NOMINAL_TOTAL_MBPS
    runs = []
    for model in ("isccp", "era5"):
        for s in REALIZATIONS:
            sched, _ = A.build_qkd_schedule(weather_seed=s, hours=12,
                                            weather_model=model)
            scale = load_mbps / nominal
            r = A.run(A.TqdConfig(horizon_s=HORIZON, seed=s, theta=0.5,
                                  load_scale=scale), sched)
            runs.append(dict(model=model, realization=s,
                             goodput_mbps=r["goodput_mbps"],
                             its_fraction=r["its_fraction"],
                             mean_aos=r["mean_aos"],
                             phys_slope=r["phys_slope"],
                             mean_virt=r["mean_virt"]))
            print(f"  {model:6s} r{s:02d} goodput {r['goodput_mbps']:6.2f} "
                  f"its {r['its_fraction']:5.3f} aos {r['mean_aos']:6.1f} "
                  f"slope {r['phys_slope']:9.2e}")
    for model in ("isccp", "era5"):
        g = [x["goodput_mbps"] for x in runs if x["model"] == model]
        sl = [x["phys_slope"] for x in runs if x["model"] == model]
        print(f"{model:6s} goodput mean {np.mean(g):6.2f} "
              f"cv {np.std(g)/np.mean(g):5.3f} | worst backlog slope "
              f"{max(sl):9.2e}")

    with open(OUT / "correlated_eta.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(OUT / "correlated_runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(runs[0].keys()))
        w.writeheader(); w.writerows(runs)
    print("wrote correlated_eta.csv, correlated_runs.csv")


if __name__ == "__main__":
    main()
