# AoS-Routing

**Age-of-Secret routing for hybrid quantum-classical tactical
non-terrestrial networks.**

Code and reproducibility artifacts accompanying:

> Liang Dong, *"Age of Secret: Joint Routing and Key Manufacturing in
> Hybrid Quantum-Classical Networks."* Journal manuscript.

This repository contains the simulator, the real Starlink Phase-1 TLE
snapshot, the derived ERA5 series, and the per-run logs behind every
number, table, and figure in the paper.  The manuscript itself is not
hosted here.

## What the paper does

A hybrid network draws on two kinds of key material that are not
interchangeable.  Q-keys arrive from satellite passes at a rate no
scheduler controls; P-keys are manufactured by running ML-KEM under a
per-node compute budget, so their rate is a control.  A block is
QKD-keyed only if *every* hop encrypted it with quantum-origin
material, which makes the two species non-fungible.

* **Secure capacity region** — a polytope in three resources, classical
  bandwidth, exogenous quantum supply, and manufactured supply under
  per-node key-establishment budgets, with a matching converse.  It
  collapses to the known quantum-only region at zero budget and when
  every class is graded, and is larger otherwise.
* **Exact decomposition** — routing is a shortest path on a transformed
  network whatever a key costs to make; manufacturing is a separate
  packing program, out of which demand gating falls rather than being
  imposed.  Because ML-KEM charges both endpoints, that program does
  not reduce to per-node knapsacks.
* **Throughput optimality under Markov-modulated supply**, which the
  persistence of weather and passes induces.
* **Provenance** — tracking the species per key unit costs no modeled
  throughput, while a policy that pools them certifies as QKD-keyed
  traffic that is not, on a fraction that grows with load.

**Simulator and policies.**  `aos_tqd.py` is the simulator for the
journal work; `aos_network.py` supplies the topology, constellation,
and pass-schedule machinery it imports.

| policy | what it is |
|---|---|
| `aos_tqd` | the algorithm of the paper: per-class shortest path on the transformed network, plus the manufacturing program |
| `tqd` | quantum-only ablation: no manufacturing, every class graded |
| `fungible` | pooled-bank baseline, identical except that it discards provenance at deposit |
| `sp` | shortest path by latency |

**Verification.**  `test_region.py` checks the claims the theory rests
on before they are asserted: that the region collapses correctly in
both limits, that it strictly enlarges otherwise, and that the greedy
manufacturing fill is optimal one-sided and badly suboptimal once both
endpoints are charged.  `region_binding.py` reports which constraints
are tight at the boundary, which is how the figures' claims about what
binds were checked rather than argued.

## Layout

```
.
├── README.md
├── LICENSE
├── .gitignore
└── sim/                                 simulator, data, and logs
    ├── data/starlink.tle                real Starlink TLE snapshot (May 2026)
    ├── data/era5_transmittance.csv      derived hourly slant-path series,
    │                                    six stations, one year
    ├── src/
    │   ├── aos_tqd.py                   the simulator: two key banks with
    │   │                                per-unit provenance, three-way edge
    │   │                                split, manufacturing program, and
    │   │                                the region LP
    │   ├── aos_network.py               topology, constellation, and
    │   │                                pass-schedule machinery
    │   ├── constellation.py             TLE loader, SGP4 visibility, key
    │   │                                rate, cloud attenuation
    │   ├── test_region.py               numerical checks on the claims the
    │   │                                theory rests on
    │   ├── region_binding.py            which constraints bind at the
    │   │                                boundary
    │   ├── region_projection.py         the region projected onto two flows
    │   ├── tqd_studies.py               load, budget, grade, freshness, and
    │   │                                manufacturing-cost sweeps
    │   ├── extra_loads.py               load points past the boundary
    │   ├── mislabel_study.py            what a pooled key bank certifies
    │   │                                wrongly
    │   ├── correlated_supply.py         reanalysis-driven weather, thirteen
    │   │                                four-week windows
    │   ├── scaling.py                   per-slot cost against network size
    │   ├── make_traces.py               per-edge mechanism traces
    │   ├── fetch_era5.py                ERA5 download (needs a CDS account)
    │   ├── era5_to_transmittance.py     condensate paths to slant-path depth
    │   ├── era5_figure.py               autocorrelation and per-site export
    │   └── make_figures_tqd.py          exports every figure's data table
    ├── results_tqd/                     the logs behind the paper
    └── figs/                            vector PDF + raster PNG figures
```

`sim/src` also retains scripts from an exploratory line that the paper
does not use: `make_figures.py`, `param_study.py`, `weight_audit.py`,
and `test_cg.py`, with their output under `results/`, `results_param/`,
and `results_lyap/`.  Nothing in the paper depends on them.

## Reproducing the experiments

Tested on Ubuntu 24.04 with Python 3.12.

```bash
pip install numpy scipy pandas matplotlib skyfield sgp4
cd sim/src

# 1. Check the claims the theory rests on
python3 test_region.py          # region limits, and where the greedy fails
python3 region_binding.py       # which constraints bind at the boundary

# 2. The studies behind the figures
python3 tqd_studies.py          # load, budget, grade, freshness, cost sweeps
python3 extra_loads.py          # load points past the boundary
python3 mislabel_study.py       # what a pooled key bank certifies wrongly
python3 region_projection.py    # the region projected onto two flows
python3 correlated_supply.py    # reanalysis-driven weather, thirteen windows
python3 scaling.py              # per-slot cost against network size
python3 make_traces.py          # per-edge mechanism traces

# 3. Export every figure's data table
python3 make_figures_tqd.py
```

The studies parallelize over seeds and take roughly an hour in total on
a single machine, most of it the twelve-hour simulated horizons; the
SGP4 schedule build dominates start-up.  No GPU is required, and every
result is deterministic given the seed.

The ERA5 series in `sim/data` is already derived and committed, so the
weather results reproduce without a download.  To rebuild it from
source you need a free Copernicus Climate Data Store account:

```bash
python3 fetch_era5.py             # ~7.6 MB of netCDF, not redistributed here
python3 era5_to_transmittance.py  # condensate paths to slant-path depth
python3 era5_figure.py            # autocorrelation and per-site statistics
```

## Real-data anchors

| Component | Source | File |
|---|---|---|
| Constellation | Real Starlink Phase-1 shell from the CelesTrak/Space-Track May 2026 catalog (incl. 52.5–53.5°, alt. 530–570 km) | `sim/data/starlink.tle`, 1306 satellites, epoch 2026-day-144 |
| QKD rate | Chen et al., *Nature* 589:214, 2021: 47.8 kbps over a typical 364 s Micius pass, used as the zenith rate and then reduced by $\cos^2 z$ and a duty factor | `qkd_rate_bps` in `constellation.py` |
| Cloud attenuation | Beer–Lambert, ISCCP mid-latitude climatology (Rossow & Schiffer 1999) with the ITU-R P.1814 framework | `cloud_transmittance` in `constellation.py` |
| Weather, measured | ERA5 reanalysis at the six station coordinates, one year hourly, converted to slant-path depth as a pencil beam rather than a grid-box mean | `sim/data/era5_transmittance.csv` |
| Compute budget | An assumption, not a measurement: one core's worth at the order of magnitude Open Quantum Safe reports, which predates FIPS 203 and cannot pin a constant for ML-KEM-768. The paper sweeps it over a factor of 64 for that reason | `node_budget_units` in `aos_tqd.py` |
| Ground stations | Real coordinates (Waco-TX, Fort Bragg-NC, Ramstein-DE, Yokota-JP, Diego Garcia, Camp Lemonnier) | `DEFAULT_GROUND_STATIONS` in `constellation.py` |

## License

Source code (`sim/src/*.py`), data (`sim/data/starlink.tle`),
configuration, and figures are released under the MIT License — see
[`LICENSE`](LICENSE).

The Starlink TLE snapshot in `sim/data/starlink.tle` is public orbital
data redistributed from the CelesTrak GP catalog
(https://celestrak.org/) and the U.S. Space-Track service.

## Citing

If you use this code or data, please cite:

```bibtex
@unpublished{dong_aos_2026,
  author = {Liang Dong},
  title  = {{Age of Secret: Joint Routing and Key Manufacturing in
            Hybrid Quantum-Classical Networks}},
  year   = {2026},
  note   = {Submitted}
}
```

## Contact

Liang Dong, Department of Electrical and Computer Engineering,
Baylor University, Waco, TX 76798, USA.
Email: `liangdng@gmail.com`.
