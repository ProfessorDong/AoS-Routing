# AoS-Routing

**Age-of-Secret routing for hybrid quantum-classical tactical
non-terrestrial networks.**

Code and data accompanying:

> Liang Dong, *"Age-of-Secret Routing for Hybrid Quantum-Classical
> Tactical Non-Terrestrial Networks,"* MILCOM 2026 (under review).

This repository reproduces every reported number, table, and figure
in the paper from the simulator and the real Starlink Phase-1 TLE
snapshot included under `sim/data/`.

## Highlights

* **Age of Secret (AoS)** — a routing-layer metric for cryptographic
  freshness and key-pool depletion risk in tactical NTNs.
* **Throughput-optimality theorem** — AoS-aware backpressure
  mean-rate-stabilizes the joint queue-and-key-deficit process for
  every arrival rate strictly inside a secure capacity region
  $\Lambda_S$ characterized in the paper.
* **Real-data simulator** — driven by a real Starlink Phase-1 TLE
  snapshot (1306 LEO satellites, May 2026 CelesTrak/Space-Track),
  SGP4 pass-window propagation, a Beer–Lambert cloud-attenuation
  model parametrized by ISCCP mid-latitude climatology, and the
  liboqs ML-KEM-768 throughput point for the PQC refresh rate.
* **Empirical winner on Age of Secret** — across 5 seeds × 5
  tactical scenarios, AoS-aware backpressure attains the lowest mean
  AoS in every scenario (2.5–29 s) while matching the secure
  goodput of the best unprincipled baseline; the closest competitor
  is 1.8–11.6× worse on AoS.

## Layout

```
.
├── main.tex / main.pdf       MILCOM 2026 manuscript
├── references.bib            BibTeX database
├── fig_aos_*.pdf             figures copied from sim/figs for the paper build
├── fig_lyapunov.pdf
├── IEEEtran.cls / .bst       IEEE conference LaTeX style (LPPL)
└── sim/
    ├── data/starlink.tle     real Starlink TLE snapshot (May 2026)
    ├── src/
    │   ├── constellation.py  real-TLE loader, SGP4 visibility,
    │   │                     QKD rate (Liao 2017 calibrated),
    │   │                     ISCCP cloud-attenuation model
    │   ├── aos_network.py    discrete-event NTN simulator,
    │   │                     six schedulers (AoS-BP, AoS-BP-Ideal,
    │   │                     and four baselines)
    │   └── make_figures.py   generates fig_*.pdf from sim/results/
    ├── results/              per-run CSVs + master.csv (sweep output)
    ├── results_lyap/         extended 3600-cycle run for Fig. 5
    └── figs/                 vector PDF + raster PNG figures
```

## Reproducing the paper

Tested on Ubuntu 24.04 with Python 3.12 and TeX Live 2023.

```bash
# 1. Install Python deps
pip install numpy scipy pandas matplotlib skyfield sgp4

# 2. Run the full simulator sweep (5 seeds × 5 scenarios × 6 schedulers)
cd sim/src
python3 aos_network.py --horizon 600 \
    --seeds 0 1 2 3 4 \
    --schedulers shortest_path pqc_only qkd_only \
                 key_rate_aware aos_backpressure aos_ideal \
    --scenarios nominal weather relay_compromise \
                traffic_surge coalition_partition \
    --out results

# 3. Extended run for the Lyapunov figure
python3 aos_network.py --horizon 3600 --seeds 0 \
    --schedulers shortest_path aos_ideal aos_backpressure \
    --scenarios nominal --out results_lyap

# 4. Regenerate figures and headline table
python3 make_figures.py

# 5. Build the PDF
cd ../..
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Total reproduction wall time on a single CPU is under 10 min (≈ 3 min
of SGP4 schedule build + ≈ 4 min of simulator sweep + the LaTeX
build). No GPU required.

## Real-data anchors

| Component | Source | File / citation |
|---|---|---|
| Constellation | Real Starlink Phase-1 shell, filtered from CelesTrak/Space-Track May 2026 catalog (incl. 52.5–53.5°, alt. 530–570 km) | `sim/data/starlink.tle` (1306 satellites, epoch 2026-day-144) |
| QKD rate model | Calibrated to Liao et al., *Nature* 549:43, 2017 (Micius decoy-state BB84, Table 1) | `qkd_rate_bps` in `constellation.py` |
| Cloud attenuation | Beer–Lambert with ISCCP mid-latitude climatology (Rossow & Schiffer 1999) and the ITU-R P.1814 FSO link framework | `cloud_transmittance` in `constellation.py` |
| PQC throughput | liboqs ML-KEM-768 single-core benchmark, network-overhead-adjusted | `Edge.pqc_refresh_bps = 200_000` in `aos_network.py` |
| Ground stations | Real coordinates (Waco-TX, Fort Bragg-NC, Ramstein-DE, Yokota-JP, Diego-Garcia, Camp Lemonnier) | `DEFAULT_GROUND_STATIONS` in `constellation.py` |

## License

* Source code (`sim/src/*.py`), data (`sim/data/starlink.tle`),
  configuration, figures, and the manuscript LaTeX/BibTeX sources:
  MIT License — see [`LICENSE`](LICENSE).
* `IEEEtran.cls` and `IEEEtran.bst` are distributed under the LaTeX
  Project Public License (LPPL); they are included unmodified from
  the official IEEE conference template.
* The Starlink TLE snapshot in `sim/data/starlink.tle` is public
  data redistributed from the CelesTrak GP catalog
  (https://celestrak.org/) and the U.S. Space-Track service.

## Citing

If you use this code or data, please cite:

```bibtex
@inproceedings{dong_aos_routing_2026,
  author    = {Liang Dong},
  title     = {{Age-of-Secret Routing for Hybrid Quantum-Classical
               Tactical Non-Terrestrial Networks}},
  booktitle = {Proc. IEEE Military Communications Conference (MILCOM)},
  year      = {2026},
  note      = {Under review}
}
```

## Contact

Liang Dong, Department of Electrical and Computer Engineering,
Baylor University, Waco, TX 76798, USA.
Email: `Liang_Dong@baylor.edu`.
