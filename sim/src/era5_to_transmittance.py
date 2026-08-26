"""Turn ERA5 cloud fields into slant-path transmittance, and measure the
two correlations that the independent-draw model gets wrong.

The simulator's cloud model has until now drawn a lognormal cloud
optical depth independently for every pass.  That is wrong in two ways
which pull in the same direction, both flattering to the network.
Consecutive passes over one site are not independent, because a frontal
system persists for many hours, so an outage is longer than the model
says.  And sites are not independent of one another at continental
separation, so the site diversity a network can exploit is smaller than
the model says.  This script measures both from reanalysis rather than
assuming either.

CLOUD OPTICAL DEPTH.  ERA5 reports grid-box mean condensate paths, and
we convert them geometrically,

    tau_cloud = 3 * LWP / (2 * rho_w * r_liq) + 3 * IWP / (2 * rho_i * r_ice)

with r_liq = 10 um and r_ice = 30 um, which are conventional effective
radii for stratiform cloud.  These reduce to tau = 150*LWP + 54.5*IWP
with the paths in kg per square meter.  The condensate paths are
grid-box means, so this tau is the mean column depth over the cell; the
cloud fraction is needed to turn it into what a beam sees, and the next
paragraph does that.

TRANSMITTANCE.  Beer-Lambert applies along the beam,

    W(theta) = exp( -tau / sin(theta) ),   tau_clear = 0.30 at 800 nm,

but the grid-box MEAN optical depth must not be substituted for tau.  A
quarter-degree cell is hundreds of times wider than a QKD beam, and
transmittance is convex in depth, so averaging the depth over the cell
and then exponentiating is a Jensen error in the pessimistic direction:
a half-cloudy hour looks opaque when a beam through the clear half
would get through untouched.  At these six sites the error is large,
turning a usable-hour fraction of 0.60 to 0.94 into 0.13 to 0.60.

We therefore treat the beam as intersecting cloud with probability tcc
and clear air otherwise, recovering the in-cloud depth by dividing the
excess over tau_clear by the fraction.  Both statistics are reported
below so the size of the difference stays visible.

Run after fetch_era5.py:  python3 era5_to_transmittance.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TAU_CLEAR = 0.30
K_LIQ = 3.0 / (2.0 * 1000.0 * 10e-6)     # 150 m^2/kg
K_ICE = 3.0 / (2.0 * 917.0 * 30e-6)      # 54.5 m^2/kg


def load_station(files: list[Path]) -> pd.DataFrame:
    """Concatenate the monthly files for one station.

    Opened one at a time and joined with pandas rather than through
    `open_mfdataset`, which would pull in dask for a table that fits in
    a few hundred kilobytes.
    """
    import xarray as xr
    parts = []
    for f in sorted(files):
        with xr.open_dataset(f) as ds:
            ds = ds.squeeze(drop=True)
            parts.append(pd.DataFrame({
                "time": pd.to_datetime(ds["valid_time"].values),
                "tcc": np.asarray(ds["tcc"].values, dtype=float).ravel(),
                "tclw": np.asarray(ds["tclw"].values, dtype=float).ravel(),
                "tciw": np.asarray(ds["tciw"].values, dtype=float).ravel(),
            }))
    df = (pd.concat(parts, ignore_index=True).dropna()
          .drop_duplicates(subset="time")
          .sort_values("time").reset_index(drop=True))
    df["tau"] = TAU_CLEAR + K_LIQ * df.tclw + K_ICE * df.tciw
    # Uniform column: the Jensen-biased statistic, kept for comparison.
    df["W_uniform"] = np.exp(-df.tau)
    # Pencil beam: clear with probability 1-tcc, in-cloud otherwise.
    frac = df.tcc.clip(lower=0.0, upper=1.0)
    tau_in = TAU_CLEAR + (df.tau - TAU_CLEAR).clip(lower=0.0) \
        / frac.clip(lower=1e-6)
    df["W_zenith"] = (1.0 - frac) * np.exp(-TAU_CLEAR) \
        + frac * np.exp(-tau_in)
    return df


def decorrelation_hours(x: np.ndarray, max_lag: int = 72) -> float:
    """First lag at which the autocorrelation falls below 1/e."""
    x = x - x.mean()
    denom = float(x @ x)
    if denom <= 0:
        return float("nan")
    for k in range(1, max_lag + 1):
        r = float(x[:-k] @ x[k:]) / denom
        if r < 1.0 / np.e:
            return float(k)
    return float(max_lag)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--era5", default="../data/era5")
    ap.add_argument("--out", default="../data/era5_transmittance.csv")
    args = ap.parse_args()

    root = Path(args.era5).resolve()
    groups: dict[str, list[Path]] = {}
    for f in root.glob("*.nc"):
        groups.setdefault(f.stem.rsplit("_", 1)[0], []).append(f)
    if not groups:
        raise SystemExit(f"no .nc files in {root}; run fetch_era5.py first")

    series, rows = {}, []
    for name in sorted(groups):
        df = load_station(groups[name])
        df["station"] = name
        rows.append(df[["station", "time", "tcc", "tau",
                        "W_uniform", "W_zenith"]])
        series[name] = df.set_index("time")["tau"]

    allrows = pd.concat(rows, ignore_index=True)
    out = Path(args.out).resolve()
    allrows.to_csv(out, index=False)

    print(f"{len(allrows)} station-hours over "
          f"{allrows.time.min().date()} to {allrows.time.max().date()}\n")
    print(f"{'station':16s} {'hours':>6s} {'mean tcc':>9s} {'med tau':>8s} "
          f"{'beam':>7s} {'unif':>7s} {'lag-1 r':>8s} {'decorr h':>9s}")
    for name, s in series.items():
        v = s.to_numpy()
        r1 = float(np.corrcoef(v[:-1], v[1:])[0, 1])
        sub = allrows[allrows.station == name]
        print(f"{name:16s} {len(v):6d} {sub.tcc.mean():9.3f} "
              f"{np.median(v):8.3f} "
              f"{(sub.W_zenith > 0.1).mean():7.3f} "
              f"{(sub.W_uniform > 0.1).mean():7.3f} {r1:8.3f} "
              f"{decorrelation_hours(v):9.1f}")

    joint = pd.DataFrame(series).dropna()
    names = list(joint.columns)
    if len(joint) < 24:
        print("\ntoo few overlapping hours for an inter-site comparison")
        print(f"\nwrote {out}")
        return
    # Cloud optical depth is heavy tailed, so Pearson understates
    # dependence; rank correlation and the correlation of the binary
    # usability indicator are the honest statistics here.
    usable = np.exp(-joint) > 0.1
    for label, c in (("Pearson", joint.corr()),
                     ("Spearman rank", joint.corr(method="spearman")),
                     ("usable-state", usable.astype(float).corr())):
        off = c.to_numpy()[~np.eye(len(names), dtype=bool)]
        print(f"\ninter-site {label} correlation over "
              f"{len(joint)} common hours: "
              f"mean {off.mean():+.3f}, max {off.max():+.3f}")
        if label == "Spearman rank":
            print("                 " + " ".join(f"{n[:7]:>8s}" for n in names))
            for n in names:
                print(f"{n:16s} "
                      + " ".join(f"{c.loc[n,m]:8.3f}" for m in names))
    print("\nThe independent-draw model asserts that every off-diagonal "
          "entry and every lag-1 entry above is zero.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
