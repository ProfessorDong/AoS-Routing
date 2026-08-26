"""Download ERA5 cloud fields at the six ground-station coordinates.

WHY.  The simulator currently draws per-pass cloud attenuation
independently from an ISCCP climatology.  Real frontal systems
correlate consecutive passes over one site, and at continental
separations they correlate sites with one another, so independent draws
overstate the site diversity a network can exploit.  ERA5 gives an
hourly reanalysis at 0.25 degrees from which both correlations can be
measured rather than assumed, and it is also what lets the
Markov-modulated supply hypothesis of the manuscript be exercised
against a real environment process instead of a synthetic chain.

ONE-TIME SETUP

  1. Register (free) at https://cds.climate.copernicus.eu and log in.

  2. Open the dataset page

       https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels

     and accept the license at the bottom of the "Download" tab.  The
     request fails with a 403 until this is done, once per account.

  3. Copy your key from https://cds.climate.copernicus.eu/profile and
     write it to ~/.cdsapirc:

       url: https://cds.climate.copernicus.eu/api
       key: <your-personal-access-token>

     Note the endpoint has no "/v2" suffix; that was the pre-2024
     address and it no longer resolves.

  4. pip install "cdsapi>=0.7.4" xarray netCDF4

THEN

  python3 fetch_era5.py --start 2025-06 --end 2026-05 --out ../data/era5

Each station is requested as its own quarter-degree box, which is a few
hundred kilobytes per station-year rather than the tens of gigabytes a
global field would be.  Requests are queued server-side and take a few
minutes each, so they are issued concurrently; the Climate Data Store
admits a small number of simultaneous requests per account and queues
the rest, which is why --workers defaults to 4 rather than to the
number of requests.  Months nearest the TLE epoch are requested first,
so an interrupted run still leaves the most useful window complete.

WHAT IS FETCHED

  total_cloud_cover                 tcc,  fraction 0-1
  total_column_cloud_liquid_water   tclw, kg m^-2
  total_column_cloud_ice_water      tciw, kg m^-2
  total_column_water_vapour         tcwv, kg m^-2

tcc alone is a poor predictor of optical attenuation because a thin
cirrus deck and a cumulonimbus tower both report near unity; tclw and
tciw are what set the slant-path optical depth, which is why all four
are requested.

Author: Liang Dong.
"""
from __future__ import annotations

import argparse
import datetime as dt
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from constellation import DEFAULT_GROUND_STATIONS

VARIABLES = [
    "total_cloud_cover",
    "total_column_cloud_liquid_water",
    "total_column_cloud_ice_water",
    "total_column_water_vapour",
]
HALF_CELL = 0.125          # ERA5 native grid is 0.25 degrees


def months(start: str, end: str):
    y0, m0 = (int(x) for x in start.split("-"))
    y1, m1 = (int(x) for x in end.split("-"))
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-06", help="YYYY-MM inclusive")
    ap.add_argument("--end", default="2026-05", help="YYYY-MM inclusive")
    ap.add_argument("--out", default="../data/era5")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    import cdsapi
    client = cdsapi.Client(quiet=True, progress=False)
    lock = threading.Lock()
    done = [0]

    jobs = []
    for gs in DEFAULT_GROUND_STATIONS:
        for y, m in months(args.start, args.end):
            target = out / f"{gs.name}_{y}{m:02d}.nc"
            if not target.exists():
                jobs.append((gs, y, m, target))
    # nearest the TLE epoch first, so an interrupted run is still useful
    jobs.sort(key=lambda j: -(j[1] * 12 + j[2]))
    print(f"{len(jobs)} requests, {args.workers} at a time")

    def fetch(job):
        gs, y, m, target = job
        ndays = (dt.date(y + (m == 12), (m % 12) + 1, 1)
                 - dt.date(y, m, 1)).days
        req = {
            "product_type": ["reanalysis"],
            "variable": VARIABLES,
            "year": [str(y)],
            "month": [f"{m:02d}"],
            "day": [f"{d:02d}" for d in range(1, ndays + 1)],
            "time": [f"{h:02d}:00" for h in range(24)],
            # area is [North, West, South, East]
            "area": [round(gs.lat_deg + HALF_CELL, 3),
                     round(gs.lon_deg - HALF_CELL, 3),
                     round(gs.lat_deg - HALF_CELL, 3),
                     round(gs.lon_deg + HALF_CELL, 3)],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        tmp = target.with_suffix(".part")
        client.retrieve("reanalysis-era5-single-levels", req, str(tmp))
        tmp.rename(target)
        with lock:
            done[0] += 1
            print(f"  [{done[0]}/{len(jobs)}] {target.name}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch, j): j for j in jobs}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as ex:
                gs, y, m, _ = futs[f]
                print(f"  FAILED {gs.name} {y}-{m:02d}: "
                      f"{type(ex).__name__}: {ex}", flush=True)

    print(f"\nDone.  Files in {out}")
    print("Next: `python3 era5_to_transmittance.py` converts these to "
          "per-station hourly slant-path transmittance and reports the "
          "lag-1 autocorrelation and the inter-site correlation matrix, "
          "which are the two numbers the independent-draw model gets "
          "wrong.")


if __name__ == "__main__":
    main()
