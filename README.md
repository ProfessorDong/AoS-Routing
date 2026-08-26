# AoS-Routing

**Age-of-Secret routing for hybrid quantum-classical tactical
non-terrestrial networks.**

Code and reproducibility artifacts accompanying:

> Liang Dong, *"Age of Secret: Throughput-Optimal Routing under
> Stochastic Key Supply in Hybrid Quantum-Classical Networks."*
> Journal manuscript in preparation.
>
> *Supersedes the MILCOM 2026 conference submission of the same
> project.*

This repository contains the simulator, the real Starlink Phase-1 TLE
snapshot, and the per-run logs that reproduce every reported number,
table, and figure.  The manuscript itself is not hosted here.

## Highlights

* **Age of Secret (AoS)** — a routing-layer metric for cryptographic
  freshness and key-pool depletion risk in tactical NTNs.
* **Secure capacity region** — a converse showing no policy is rate
  stable outside $\Lambda_S$ for any key-buffer size, and a matching
  achievability result inside it.
* **Exact per-slot scheduling** — the per-slot max-weight problem over
  path actions is a linear program, solved exactly by column
  generation whose pricing subproblem is one Dijkstra run per flow on
  the drift-derived edge cost plus the master's dual price.  The
  per-flow greedy that omits the dual price forgoes an unbounded
  fraction of the objective; `test_cg.py` exhibits the counterexample
  and checks the solver against full path enumeration.
* **Real-data simulator** — driven by a real Starlink Phase-1 TLE
  snapshot (1306 LEO satellites, May 2026 CelesTrak/Space-Track),
  SGP4 pass-window propagation, a Beer–Lambert cloud-attenuation
  model parametrized by ISCCP mid-latitude climatology, and the
  liboqs ML-KEM-768 throughput point for the PQC refresh rate.
* **Empirical results** — across 20 seeds × 5 tactical scenarios,
  AoS-BP attains 2.2–40.2 s mean Age of Secret while matching the
  secure goodput of the best baseline to within 0.01 %. The strongest
  unprincipled competitor (key-rate-aware) sits at 21–42 s, and
  PQC-only at 61–94 s. QKD-only reaches competitive freshness only
  by delivering a fifth of the offered load.

**Scheduler naming.**

| key | paper name | status |
|---|---|---|
| `aos_cg` | **AoS-BP** | main algorithm; solves the per-slot max-weight LP exactly by column generation |
| `aos_greedy` | AoS-BP-G | ablation; per-flow greedy, unbounded loss (Proposition 1) |
| `aos_backpressure` | AoS-BP-H | ablation; cost-minimizing heuristic from the MILCOM version |
| `aos_ideal` | AoS-BP-Ideal | the per-edge max-weight scheduler of the achievability theorem |
| `shortest_path`, `pqc_only`, `qkd_only`, `key_rate_aware` | baselines | Dijkstra over scheduler-specific edge weights |

## Layout

```
.
├── README.md
├── LICENSE
├── .gitignore
└── sim/                                    simulator, data, and logs
    ├── data/starlink.tle                   real Starlink TLE snapshot (May 2026)
    ├── src/
    │   ├── constellation.py                real-TLE loader, SGP4 visibility,
    │   │                                   QKD rate (Liao 2017 calibrated),
    │   │                                   ISCCP cloud-attenuation model
    │   ├── aos_network.py                  discrete-event NTN simulator,
    │   │                                   eight schedulers (AoS-BP, two
    │   │                                   ablations, AoS-BP-Ideal, four
    │   │                                   baselines)
    │   ├── test_cg.py                      correctness tests for the column-
    │   │                                   generation solver: the greedy
    │   │                                   counterexample, exactness against
    │   │                                   full path enumeration, feasibility
    │   ├── param_study.py                  parameter-selection and sensitivity
    │   │                                   study (weight form, K_max, T_a,
    │   │                                   load sweep, penalty-backlog curve,
    │   │                                   exact-vs-greedy gap)
    │   ├── weight_audit.py                 decomposition of the routing weight
    │   │                                   and the omega sweep
    │   └── make_figures.py                 generates figures from sim/results/
    ├── results/                            per-run CSVs + master.csv (sweep output)
    ├── results_param/                      parameter-study output
    ├── results_lyap/                       extended 3600-cycle run for the
    │                                       Lyapunov verification figure
    └── figs/                               vector PDF + raster PNG figures
```

`sim/` now carries the journal version of the code; the MILCOM
scheduler is retained inside it as the `aos_backpressure` ablation
rather than in a separate tree.

## Reproducing the experiments

Tested on Ubuntu 24.04 with Python 3.12.

```bash
# 1. Install Python dependencies
pip install numpy scipy pandas matplotlib skyfield sgp4

# 2. Check the scheduler solves the per-slot problem exactly
cd sim/src
python3 test_cg.py

# 3. Run the full simulator sweep (20 seeds × 5 scenarios × 8 schedulers)
python3 aos_network.py --horizon 600 \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --schedulers shortest_path pqc_only qkd_only key_rate_aware \
                 aos_backpressure aos_greedy aos_cg aos_ideal \
    --scenarios nominal weather relay_compromise \
                traffic_surge coalition_partition \
    --out results

# 4. Extended run for the Lyapunov verification figure
python3 aos_network.py --horizon 3600 --seeds 0 \
    --schedulers shortest_path aos_ideal aos_backpressure \
                 aos_greedy aos_cg \
    --scenarios nominal --out results_lyap

# 5. Parameter-selection and sensitivity study
python3 param_study.py

# 6. Routing-weight decomposition and the omega sweep
python3 weight_audit.py

# 7. Regenerate figures and headline table
python3 make_figures.py
```

Total reproduction wall time on a single CPU is roughly 40 min, most of
it the SGP4 schedule build (20 seeds × 1306 satellites × 6 ground
stations) and the parameter study. The main sweep itself is about
6 min. No GPU required; every result is deterministic given
`PYTHONHASHSEED=0`.

## Real-data anchors

| Component | Source | File / citation |
|---|---|---|
| Constellation | Real Starlink Phase-1 shell, filtered from CelesTrak/Space-Track May 2026 catalog (incl. 52.5–53.5°, alt. 530–570 km) | `sim/data/starlink.tle` (1306 satellites, epoch 2026-day-144) |
| QKD rate model | Calibrated to Liao et al., *Nature* 549:43, 2017 (Micius decoy-state BB84, Table 1) | `qkd_rate_bps` in `constellation.py` |
| Cloud attenuation | Beer–Lambert with ISCCP mid-latitude climatology (Rossow & Schiffer 1999) and the ITU-R P.1814 FSO link framework | `cloud_transmittance` in `constellation.py` |
| PQC throughput | liboqs ML-KEM-768 single-core benchmark, network-overhead-adjusted | `Edge.pqc_refresh_bps = 200_000` in `aos_network.py` |
| Ground stations | Real coordinates (Waco-TX, Fort Bragg-NC, Ramstein-DE, Yokota-JP, Diego-Garcia, Camp Lemonnier) | `DEFAULT_GROUND_STATIONS` in `constellation.py` |

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
@unpublished{dong_aos_routing_2026,
  author = {Liang Dong},
  title  = {{Age of Secret: Throughput-Optimal Routing under Stochastic
            Key Supply in Hybrid Quantum-Classical Networks}},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

## Contact

Liang Dong, Department of Electrical and Computer Engineering,
Baylor University, Waco, TX 76798, USA.
Email: `liangdng@gmail.com`.
