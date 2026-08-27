"""
Parameter-selection and sensitivity study for AoS-Routing.

This script reproduces the study that fixed the Algorithm 1 routing weight,
the key-buffer cap K_max, and the Eq. (2) age cap T_a.  It exists because
those three choices are coupled, and because an earlier draft's
headline result turned out to depend on an unstated coincidence between
alpha and K_max: the shipped key term alpha*(K_max - K) is in absolute
key-bits, so its magnitude rescales with the buffer size while the latency
and AoS terms do not.  At large K_max it swamps the AoS term and Algorithm 1
collapses into the Key-rate-aware baseline.

Four studies, in the order they must be run:

  A. weight_variant x K_max, at two horizons.  Selects the form of the
     Eq. (8) key-scarcity term on a pre-specified invariance criterion.
  B. K_max sensitivity at the selected variant, reporting overflow measured
     on traffic-carrying edges.  A permanently idle edge saturates at any
     finite K_max given a long enough horizon, so whole-network overflow is
     not a well-posed test of Theorem 1 assumption (iii).
  C. T_a sensitivity.
  D. Load sweep, to locate the empirical stability boundary of Lambda_S.
  E. K_max invariance of AoS-BP (Algorithm 1').  Its edge cost is the
     drift-derived psi_e = omega*rho*Z_e + beta'*L_e + chi'*AoS_e, and Z
     carries no K_max dependence, so the theory predicts *exact*
     invariance.  This study tests that prediction.
  F. The utility-delay curve in chi', the drift-plus-penalty V knob.

Selection rules, fixed before any variant was run:

  * Each candidate key term is calibrated to equal the shipped W0 term at
    K = K_max/2, using the reference configuration (K_max = 5 Mb, a 1 Gbps
    QKD-capable edge).  No free fitting.
  * Primary criterion: scale invariance, max/min of mean AoS across the
    K_max grid.  Secondary: mean AoS over the grid.  Constraint: goodput
    within 1% of the best variant.

Author: Liang Dong.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import aos_network as A

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results_param"

K_MAX_GRID = [5e6, 5e7, 5e8, 5e9]
VARIANTS = ["W0", "W1", "W2", "W3"]
SELECTED_VARIANT = "W3"
# Algorithm 1' does not use a weight_variant at all: its edge cost is the
# drift-derived psi_e = omega*rho*Z_e + beta'*L_e + chi'*AoS_e.  Study E
# checks the resulting prediction of exact K_max invariance, and study F
# traces the utility-delay curve in chi'.


def _mean(rows, field):
    return float(np.mean([r[field] for r in rows]))


def _run(schedules, seeds, **kw):
    return [A.run(A.SimConfig(**dict(kw, seed=s)), schedules[s], OUT)
            for s in seeds]


def study_a(schedules, seeds, writer):
    """weight_variant x K_max x horizon."""
    for horizon in (600, 3600):
        hs = seeds if horizon == 600 else seeds[:1]
        for var in VARIANTS:
            for km in K_MAX_GRID:
                r = _run(schedules, hs, horizon_s=horizon,
                         scheduler="aos_backpressure", scenario="nominal",
                         k_max_bits=km, weight_variant=var)
                writer.writerow(dict(
                    study="A", horizon_s=horizon, weight_variant=var,
                    k_max_bits=km, aos_T_a=300.0, load_scale=1.0,
                    scheduler="aos_backpressure", n_seeds=len(hs),
                    mean_aos=_mean(r, "mean_aos"),
                    goodput_mbps=_mean(r, "secure_goodput_bps") / 1e6,
                    active_overflow_frac=_mean(r, "active_overflow_frac")))


def study_b(schedules, seeds, writer):
    """K_max sensitivity at the selected variant, all reference schedulers."""
    for horizon in (600, 3600):
        hs = seeds if horizon == 600 else seeds[:1]
        for km in K_MAX_GRID:
            for sched in ("aos_backpressure", "key_rate_aware", "qkd_only",
                          "shortest_path"):
                r = _run(schedules, hs, horizon_s=horizon, scheduler=sched,
                         scenario="nominal", k_max_bits=km,
                         weight_variant=SELECTED_VARIANT)
                writer.writerow(dict(
                    study="B", horizon_s=horizon,
                    weight_variant=SELECTED_VARIANT, k_max_bits=km,
                    aos_T_a=300.0, load_scale=1.0, scheduler=sched,
                    n_seeds=len(hs), mean_aos=_mean(r, "mean_aos"),
                    goodput_mbps=_mean(r, "secure_goodput_bps") / 1e6,
                    active_overflow_frac=_mean(r, "active_overflow_frac")))


def study_c(schedules, seeds, writer):
    """T_a sensitivity."""
    for ta in (120.0, 300.0, 600.0, 1200.0):
        for sched in ("aos_backpressure", "key_rate_aware", "shortest_path"):
            r = _run(schedules, seeds, horizon_s=600, scheduler=sched,
                     scenario="nominal", aos_T_a=ta,
                     weight_variant=SELECTED_VARIANT)
            writer.writerow(dict(
                study="C", horizon_s=600, weight_variant=SELECTED_VARIANT,
                k_max_bits=A.Edge.k_max_bits, aos_T_a=ta, load_scale=1.0,
                scheduler=sched, n_seeds=len(seeds),
                mean_aos=_mean(r, "mean_aos"),
                goodput_mbps=_mean(r, "secure_goodput_bps") / 1e6,
                active_overflow_frac=_mean(r, "active_overflow_frac")))


def study_d(schedules, writer, horizon: int = 1800):
    """Load sweep.  Classifies stability by the OLS slope of the total queue
    backlog over the second half of the run, which is the finite-horizon
    stand-in for the strong-stability conclusion."""
    for ls in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
        for sched in ("aos_cg", "aos_greedy", "aos_backpressure",
                      "aos_ideal", "shortest_path"):
            cfg = A.SimConfig(horizon_s=horizon, seed=0, scheduler=sched,
                              scenario="nominal", load_scale=ls,
                              weight_variant=SELECTED_VARIANT)
            nodes, edges = A.default_topology()
            net = A.AoSNetwork(nodes, edges, A.default_flows(), cfg,
                               schedules[0])
            for t in range(horizon):
                net.step(t)
            q = np.array([l.queue_total_bits for l in net.logs], dtype=float)
            half = horizon // 2
            slope = float(np.polyfit(np.arange(half, horizon), q[half:], 1)[0])
            writer.writerow(dict(
                study="D", horizon_s=horizon,
                weight_variant=SELECTED_VARIANT,
                k_max_bits=A.Edge.k_max_bits, aos_T_a=cfg.aos_T_a,
                load_scale=ls, scheduler=sched, n_seeds=1,
                mean_aos=float(np.mean([l.aos_mean for l in net.logs])),
                goodput_mbps=sum(l.total_secure_bits for l in net.logs)
                             / horizon / 1e6,
                active_overflow_frac=float("nan"),
                queue_slope_bits_per_cycle=slope))


def study_e(schedules, seeds, writer):
    """K_max invariance of AoS-BP.  Theory predicts exact invariance."""
    for horizon in (600, 3600):
        hs = seeds if horizon == 600 else seeds[:1]
        for km in K_MAX_GRID:
            r = _run(schedules, hs, horizon_s=horizon,
                     scheduler="aos_cg", scenario="nominal",
                     k_max_bits=km)
            writer.writerow(dict(
                study="E", horizon_s=horizon, weight_variant="drift",
                k_max_bits=km, aos_T_a=300.0, load_scale=1.0,
                scheduler="aos_cg", n_seeds=len(hs),
                mean_aos=_mean(r, "mean_aos"),
                goodput_mbps=_mean(r, "secure_goodput_bps") / 1e6,
                active_overflow_frac=_mean(r, "active_overflow_frac")))


def study_f(schedules, seeds, writer):
    """Utility-delay curve in chi' (the V knob of the tradeoff theorem)."""
    for chi in (5e-3, 5e0, 5e2, 5e3, 5e4, 5e5, 5e6, 5e7):
        r = _run(schedules, seeds, horizon_s=600, scheduler="aos_cg",
                 scenario="nominal", bp_chi_prime=chi)
        writer.writerow(dict(
            study="F", horizon_s=600, weight_variant="drift",
            k_max_bits=A.Edge.k_max_bits, aos_T_a=300.0, load_scale=1.0,
            scheduler="aos_cg", n_seeds=len(seeds), chi_prime=chi,
            mean_aos=_mean(r, "mean_aos"),
            goodput_mbps=_mean(r, "secure_goodput_bps") / 1e6,
            mean_queue_bits=_mean(r, "mean_queue"),
            active_overflow_frac=_mean(r, "active_overflow_frac")))


def study_g(schedules, seeds, writer):
    """Cost of the exact per-slot solve, and what it buys, against load.

    Three quantities per operating point: the mean number of column-
    generation pricing rounds (one Dijkstra per flow per round), the
    fraction of the exact max-weight objective the per-flow greedy
    forgoes, and the worst such fraction over the run.  The greedy has no
    bounded-loss guarantee, so the point of the sweep is to find where
    the absent guarantee starts to cost something measurable.
    """
    for ls in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0):
        r = _run(schedules, seeds, horizon_s=600, scheduler="aos_cg",
                 scenario="nominal", load_scale=ls)
        g = _run(schedules, seeds, horizon_s=600, scheduler="aos_greedy",
                 scenario="nominal", load_scale=ls)
        writer.writerow(dict(
            study="G", horizon_s=600, weight_variant="drift",
            k_max_bits=A.Edge.k_max_bits, aos_T_a=300.0, load_scale=ls,
            scheduler="aos_cg", n_seeds=len(seeds),
            mean_aos=_mean(r, "mean_aos"),
            goodput_mbps=_mean(r, "secure_goodput_bps") / 1e6,
            mean_queue_bits=_mean(r, "mean_queue"),
            cg_mean_iters=_mean(r, "cg_mean_iters"),
            cg_gap_frac=_mean(r, "cg_gap_frac"),
            cg_gap_max=_mean(r, "cg_gap_max"),
            cg_nonconverged=_mean(r, "cg_nonconverged"),
            greedy_aos=_mean(g, "mean_aos"),
            greedy_goodput_mbps=_mean(g, "secure_goodput_bps") / 1e6,
            greedy_queue_bits=_mean(g, "mean_queue")))


def main():
    seeds = [0, 1, 2]
    OUT.mkdir(parents=True, exist_ok=True)
    print("Building per-seed QKD pass schedules")
    schedules = {s: A.build_qkd_schedule(weather_seed=s, hours=12)[0]
                 for s in seeds}
    fields = ["study", "horizon_s", "weight_variant", "k_max_bits", "aos_T_a",
              "load_scale", "scheduler", "n_seeds", "mean_aos",
              "goodput_mbps", "active_overflow_frac",
              "queue_slope_bits_per_cycle", "chi_prime", "mean_queue_bits",
              "cg_mean_iters", "cg_gap_frac", "cg_gap_max",
              "cg_nonconverged", "greedy_aos", "greedy_goodput_mbps",
              "greedy_queue_bits"]
    with open(OUT / "param_study.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for name, fn in (("A weight-variant selection", study_a),
                         ("B K_max sensitivity", study_b),
                         ("C T_a sensitivity", study_c),
                         ("E K_max invariance of AoS-BP", study_e),
                         ("F penalty-backlog curve", study_f),
                         ("G exact solve vs greedy", study_g)):
            print(f"  study {name}")
            fn(schedules, seeds, w)
        print("  study D load sweep")
        study_d(schedules, w)
    print(f"Wrote {OUT / 'param_study.csv'}")


if __name__ == "__main__":
    main()
