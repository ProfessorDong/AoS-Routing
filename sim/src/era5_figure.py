"""Export the reanalysis findings as pgfplots tables.

Section V-F reports three things about a year of ERA5 at the six station
coordinates, and reports them as numbers.  Two of them are better seen
than read.

  TEMPORAL.  Cloud optical depth over a site is strongly autocorrelated,
  so the independent-per-pass draw is wrong in time.  The autocorrelation
  function against lag shows this at a glance, and the independent model
  is the horizontal line at zero, which makes the gap between assumption
  and measurement the whole content of the panel.

  SPATIAL.  The sites are nearly uncorrelated with one another, which is
  what the independent draw gets right, and they are very unequal, which
  it gets wrong because it applies one distribution everywhere.  A
  per-site usable-hour fraction next to the inter-site correlation makes
  both points in one panel.

Run after era5_to_transmittance.py:  python3 era5_figure.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
OUT = SIM.parent / "journal" / "data"
MAX_LAG = 48
USABLE = 0.1

# Short labels, in the order the panel draws them.
SHORT = {"Waco-TX": "Waco", "FortBragg-NC": "Bragg",
         "RamsteinAB-DE": "Ramstein", "Yokota-JP": "Yokota",
         "Diego-Garcia": "Diego G.", "CampLemonnier": "Lemonnier"}


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - x.mean()
    denom = float(x @ x)
    if denom <= 0:
        return np.zeros(max_lag + 1)
    return np.array([1.0] + [float(x[:-k] @ x[k:]) / denom
                             for k in range(1, max_lag + 1)])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SIM / "data" / "era5_transmittance.csv",
                     parse_dates=["time"])
    sites = [s for s in SHORT if s in set(df.station)]

    # (1) autocorrelation of optical depth, per site
    series = {s: df[df.station == s].sort_values("time").tau.to_numpy()
              for s in sites}
    curves = {s: acf(v, MAX_LAG) for s, v in series.items()}
    with open(OUT / "era5_acf.dat", "w") as fh:
        fh.write("lag " + " ".join(SHORT[s].replace(" ", "") for s in sites)
                 + "\n")
        for k in range(MAX_LAG + 1):
            fh.write(f"{k} " + " ".join(f"{curves[s][k]:.6g}"
                                        for s in sites) + "\n")
    lag1 = {s: curves[s][1] for s in sites}
    print(f"  era5_acf.dat: lag-1 between {min(lag1.values()):.3f} and "
          f"{max(lag1.values()):.3f}")
    efold = {}
    for s in sites:
        below = np.where(curves[s] < 1.0 / np.e)[0]
        efold[s] = int(below[0]) if len(below) else MAX_LAG
    print(f"    e-folding {min(efold.values())} to {max(efold.values())} h")

    # (2) per-site usable-hour fraction, the heterogeneity the ISCCP
    #     climatology draw cannot express because it is one distribution
    frac = {s: float((df[df.station == s].W_zenith > USABLE).mean())
            for s in sites}
    order = sorted(sites, key=lambda s: frac[s])
    with open(OUT / "era5_sites.dat", "w") as fh:
        fh.write("idx frac label\n")
        for k, s in enumerate(order):
            fh.write(f"{k} {frac[s]:.6g} {SHORT[s].replace(' ', '')}\n")
    print(f"  era5_sites.dat: usable fraction {min(frac.values()):.3f} "
          f"({SHORT[order[0]]}) to {max(frac.values()):.3f} "
          f"({SHORT[order[-1]]})")

    # (3) inter-site rank correlation, reported as the spread of the
    #     off-diagonal entries against the zero the model asserts
    joint = pd.DataFrame({s: pd.Series(
        df[df.station == s].sort_values("time").tau.to_numpy())
        for s in sites}).dropna()
    c = joint.corr(method="spearman").to_numpy()
    off = c[~np.eye(len(sites), dtype=bool)]
    print(f"  inter-site Spearman: mean {off.mean():+.3f}, "
          f"min {off.min():+.3f}, max {off.max():+.3f}")
    with open(OUT / "era5_intersite.dat", "w") as fh:
        fh.write("mean min max\n")
        fh.write(f"{off.mean():.6g} {off.min():.6g} {off.max():.6g}\n")
    print(f"exported to {OUT}")


if __name__ == "__main__":
    main()
