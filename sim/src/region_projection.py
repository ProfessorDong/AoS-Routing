"""The secure capacity region, drawn rather than summarized.

Every figure so far reports the region as a single number: the largest
scaling of one fixed arrival vector that stays inside it.  That is what
a load sweep can be compared against, but it hides the shape, and the
shape is where the two species differ.  Manufacturing does not inflate
the region uniformly.  It buys nothing along the directions that graded
traffic occupies and a great deal along the others, so the region grows
anisotropically and a radial plot shows that directly.

We project onto two of the five flows and sweep the direction.  For each
angle the linear program of Definition 5 is solved with the other three
flows held at zero, which gives the exact radial extent of the region in
that direction; the locus over all angles is the boundary of the
two-dimensional projection.

The pair is chosen to make the asymmetry visible rather than to flatter
it.  On this topology every ground station attaches to two gateways, and
four of the five flows happen to share a gateway with their destination,
so they need two hops and never touch the mesh.  bulk-A does not:
CampLemonnier and FortBragg have no gateway in common.  Its flexible
traffic can cross the gateway mesh on manufactured keys, while its
graded traffic cannot cross the mesh at all and has to relay through a
third ground station.  Pairing it with ses-A, an ordinary two-hop flow,
puts a flow that manufacturing helps on one axis and a flow it barely
helps on the other.

Three regions are computed:

  prior work   B = 0, every class graded, so the region is the
               multicommodity flow region under min(C_e, eta_e)
  ours, 1/2    half of each flow graded
  ours, 0      no grade requirement at all

Run:  python3 region_projection.py
"""
from __future__ import annotations

import csv
import dataclasses
import math
from pathlib import Path

import numpy as np

import aos_tqd as A

OUT = Path(__file__).resolve().parent.parent / "results_tqd"
HORIZON = 43200
SEEDS = [0, 1, 2, 3, 4]
N_ANGLES = 25
PAIR = ("bulk-A", "ses-A")


def radial(eta, cfg, flows, edges, nodes, i, j, phi):
    """Largest t with t*(cos phi, sin phi) inside the region, in Mbps."""
    probe = []
    for k, f in enumerate(flows):
        g = dataclasses.replace(f)
        # the direction is expressed in Mbps, so the scaling t that
        # region_boundary returns is already in those units
        if k == i:
            g.arrival_bps = math.cos(phi) * 1e6
        elif k == j:
            g.arrival_bps = math.sin(phi) * 1e6
        else:
            g.arrival_bps = 0.0
        probe.append(g)
    t = A.region_boundary(eta, cfg, probe, edges, nodes)
    return t * math.cos(phi), t * math.sin(phi)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nodes, edges = A.default_topology()
    flows = A.scaled_flows()
    names = [f.name for f in flows]
    i, j = names.index(PAIR[0]), names.index(PAIR[1])
    print(f"projecting onto {PAIR[0]} (x) and {PAIR[1]} (y)")

    sched = {s: A.build_qkd_schedule(weather_seed=s, hours=12)[0]
             for s in SEEDS}
    etas = [A.empirical_eta(sched[s], HORIZON, edges) for s in SEEDS]
    eta = {k: float(np.mean([e[k] for e in etas])) for k in etas[0]}

    cases = [("prior", A.TqdConfig(theta=1.0, manufacture=False)),
             ("half", A.TqdConfig(theta=0.5)),
             ("flex", A.TqdConfig(theta=0.0))]
    angles = [0.5 * math.pi * k / (N_ANGLES - 1) for k in range(N_ANGLES)]
    rows = []
    for phi in angles:
        r = {"phi": phi}
        for label, cfg in cases:
            x, y = radial(eta, cfg, flows, edges, nodes, i, j, phi)
            r[f"x_{label}"], r[f"y_{label}"] = x, y
        rows.append(r)
        print(f"  phi {math.degrees(phi):5.1f} deg  "
              + "  ".join(f"{lb}=({r['x_'+lb]:6.2f},{r['y_'+lb]:6.2f})"
                          for lb, _ in cases))

    with open(OUT / "region_projection.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Area of each projected region, by the shoelace formula on the
    # sampled boundary plus the two axes.  A ratio of areas is a fairer
    # summary of "how much bigger" than any single radial scaling.
    for label, _ in cases:
        pts = [(0.0, 0.0)] + [(r[f"x_{label}"], r[f"y_{label}"])
                              for r in rows] + [(0.0, 0.0)]
        area = 0.5 * abs(sum(pts[k][0] * pts[k + 1][1]
                             - pts[k + 1][0] * pts[k][1]
                             for k in range(len(pts) - 1)))
        print(f"  area({label}) = {area:8.2f} Mbps^2")
    print("wrote region_projection.csv")


if __name__ == "__main__":
    main()
