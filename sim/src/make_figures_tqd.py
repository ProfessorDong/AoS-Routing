"""Export the hybrid-supply results as pgfplots tables.

The manuscript draws its figures natively in pgfplots so that figure
text is typeset by LaTeX in the document font.  This writes the plain
tables those figures read.  Nothing is smoothed or fitted; where a
theoretical curve is plotted alongside a measurement, the curve comes
from the linear program of `aos_tqd.region_boundary` and not from the
data.

Run after aos_tqd.py and tqd_studies.py:  python3 make_figures_tqd.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
PAPER = SIM.parent
OUT = PAPER / "journal" / "data"


def _w(name: str, header: str, rows) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w") as fh:
        fh.write(header + "\n")
        for r in rows:
            fh.write(" ".join(f"{v:.6g}" if isinstance(v, float) else str(v)
                              for v in r) + "\n")
    print(f"  wrote {name} ({len(rows)} rows)")


def main() -> None:
    st = pd.read_csv(SIM / "results_tqd" / "studies.csv")
    extra = SIM / "results_tqd" / "extra_loads.csv"
    if extra.exists():          # load points past this paper's boundary
        st = pd.concat([st, pd.read_csv(extra)], ignore_index=True)

    # (1) grade-capacity tradeoff: the reciprocal of Theorem 3, measured
    c = st[st.study == "C"].sort_values("theta")
    tqd_boundary = float(st[(st.study == "A") & (st.policy == "tqd")]
                         .lp_boundary_mbps.iloc[0])
    _w("gradecap.dat", "theta bound_ours bound_prior goodput aos",
       [(float(r.theta), float(r.lp_boundary_mbps), tqd_boundary,
         float(r.goodput_mbps), float(r.mean_aos)) for _, r in c.iterrows()])

    # (2) load sweep: where each policy's backlog lifts off the axis,
    #     against the boundary the linear program predicts
    a = st[st.study == "A"]
    ours = a[a.policy == "aos_tqd"].sort_values("offered_mbps")
    prior = a[a.policy == "tqd"].sort_values("offered_mbps")
    _w("loadsweep2.dat",
       "offered slope_ours slope_prior gp_ours gp_prior",
       [(float(o.offered_mbps), max(0.0, float(o.phys_slope)),
         max(0.0, float(p.phys_slope)), float(o.goodput_mbps),
         float(p.goodput_mbps))
        for (_, o), (_, p) in zip(ours.iterrows(), prior.iterrows())])
    print(f"    boundaries: ours {ours.lp_boundary_mbps.iloc[0]:.2f}, "
          f"prior {tqd_boundary:.2f} Mbps")

    # (3) budget sweep
    b = st[st.study == "B"].sort_values("budget_scale")
    # The knee sits below an eighth of a core, so the figure axis is
    # logarithmic and the zero-budget point is placed half a step below
    # the smallest budget tested rather than dropped.
    nz = [float(x) for x in b.budget_scale if x > 0]
    zero_at = min(nz) / 2 if nz else 0.01
    _w("budget.dat", "scale bound goodput its",
       [(float(r.budget_scale) if r.budget_scale > 0 else zero_at,
         float(r.lp_boundary_mbps),
         float(r.goodput_mbps), float(r.its_fraction))
        for _, r in b.iterrows()])

    # (4) demand gating, derived: manufacturing falls away as its price
    #     rises, long before goodput does
    e = st[st.study == "E"].sort_values("V_nu")
    _w("gating.dat", "vnu manufactured goodput its",
       [(max(float(r.V_nu), 0.1), float(r.manufacture_utilisation),
         float(r.goodput_mbps), float(r.its_fraction))
        for _, r in e.iterrows()])

    # (5) freshness weight selection
    d = st[st.study == "D"].sort_values("V_chi")
    _w("vchi.dat", "vchi aos goodput slope its",
       [(max(float(r.V_chi), 0.01), float(r.mean_aos),
         float(r.goodput_mbps), max(0.0, float(r.phys_slope)),
         float(r.its_fraction)) for _, r in d.iterrows()])

    # (6) what pooling the two species costs, against load
    mp = SIM / "results_tqd" / "mislabel.csv"
    if mp.exists():
        m = pd.read_csv(mp)
        f = m[m.policy == "fungible"]
        o = m[m.policy == "aos_tqd"]
        rows = []
        for th in sorted(f.theta.unique()):
            for _, r in f[f.theta == th].sort_values("load_scale").iterrows():
                oo = o[(o.theta == th) & (o.load_scale == r.load_scale)]
                rows.append((float(r.load_scale), float(th),
                             float(r.mislabelled_fraction),
                             float(oo.mislabelled_fraction.iloc[0]),
                             float(r.goodput_mbps),
                             float(oo.goodput_mbps.iloc[0])))
        _w("mislabel.dat", "load theta mis_pooled mis_ours gp_pooled gp_ours",
           rows)
        print(f"    pooled mislabelling {f.mislabelled_fraction.min():.3f}"
              f" to {f.mislabelled_fraction.max():.3f}; ours "
              f"{o.mislabelled_fraction.max():.3f}")

    # (7) mechanism trace on one ground-station edge: pass-driven quantum
    #     supply, manufactured supply, and how service splits between them
    tp = SIM / "results_tqd" / "trace_gs.csv"
    mp2 = SIM / "results_tqd" / "trace_mesh.csv"
    if tp.exists():
        t = pd.read_csv(tp)
        t["hour"] = t.t / 3600.0
        # The mesh edge is carried alongside so that the claim "the
        # manufactured keys go to the mesh" is shown rather than
        # asserted.  Both traces are decimated on the same time base.
        mesh = (pd.read_csv(mp2).servedP.to_numpy() if mp2.exists()
                else np.zeros(len(t)))
        if len(mesh) != len(t):
            mesh = np.zeros(len(t))
        _w("mechanism2.dat",
           "hour qkd mfg servedQ servedP meshP total age xq",
           [(float(r.hour), float(r.qkd), float(r.mfg), float(r.servedQ),
             float(r.servedP), float(mesh[k]),
             float(r.servedQ + r.servedP),
             float(r.ageQ), float(r.xq))
            for k, (_, r) in enumerate(t.iterrows())])

    # (8) correlated supply: what the reanalysis-driven weather does to
    #     the dispersion of the boundary, which the independent draw
    #     averages away
    cp = SIM / "results_tqd" / "correlated_runs.csv"
    if cp.exists():
        cr = pd.read_csv(cp)
        rows = []
        for i, r in enumerate(sorted(cr.realization.unique())):
            g = cr[cr.realization == r]
            iso = g[g.model == "isccp"]
            er = g[g.model == "era5"]
            if iso.empty or er.empty:
                continue
            rows.append((float(r),
                         float(iso.goodput_mbps.iloc[0]),
                         float(er.goodput_mbps.iloc[0]),
                         float(iso.its_fraction.iloc[0]),
                         float(er.its_fraction.iloc[0]),
                         max(0.0, float(iso.phys_slope.iloc[0])),
                         max(0.0, float(er.phys_slope.iloc[0]))))
        _w("correlated.dat",
           "realization gp_iid gp_era5 its_iid its_era5 slope_iid slope_era5",
           rows)
        for m in ("isccp", "era5"):
            g = cr[cr.model == m].goodput_mbps
            print(f"    {m}: goodput {g.mean():.2f} Mbps, "
                  f"cv {g.std()/g.mean():.3f}")

    # (9) the region itself, projected onto two flows.  Manufacturing
    #     does more than enlarge it: the per-node compute budget couples
    #     flows that the per-edge quantum supply leaves independent, so
    #     the enlarged region acquires a diagonal facet the quantum-only
    #     region does not have.
    rp = SIM / "results_tqd" / "region_projection.csv"
    if rp.exists():
        d = pd.read_csv(rp)
        _w("region.dat",
           "x_prior y_prior x_half y_half x_flex y_flex",
           [(float(r.x_prior), float(r.y_prior), float(r.x_half),
             float(r.y_half), float(r.x_flex), float(r.y_flex))
            for _, r in d.iterrows()])
        for lab in ("prior", "half", "flex"):
            x, y = d["x_" + lab].to_numpy(), d["y_" + lab].to_numpy()
            s = x + y
            binding = int((s > 0.99 * s.max()).sum())
            print(f"    {lab:5s}: x*={x.max():6.2f} y*={y.max():6.2f} "
                  f"Mbps, diagonal binding at {binding}/{len(s)} angles")

    print(f"\nexported to {OUT}")


if __name__ == "__main__":
    main()
