"""Per-edge traces for the mechanism figure.

Fig. 1 shows one ground-station edge over the whole schedule: quantum
supply arriving in pass-shaped bursts, manufactured supply carrying the
remainder, and the freshness surrogate the routing weight reads.  This
regenerates those traces, which must be rebuilt whenever the supply
model, the budget accounting, or the surrogate changes.

Two edges are traced.  A ground-station edge, which has both species,
and a gateway-mesh edge, which has eta_e = 0 exactly and is therefore
served by manufactured keys alone; the latter is the mechanism of
Theorem 2(iii) visible at the level of a single link.

Run:  python3 make_traces.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import aos_tqd as A

OUT = Path(__file__).resolve().parent.parent / "results_tqd"
HORIZON = 43200
DECIMATE = 120          # one row every two minutes, which is ample here


def trace_one(edge_key, name: str, load_scale: float = 2.0) -> None:
    nodes, edges = A.default_topology()
    flows = A.scaled_flows()
    sched, _ = A.build_qkd_schedule(weather_seed=0, hours=12)
    net = A.HybridNetwork(nodes, edges, flows,
                          A.TqdConfig(horizon_s=HORIZON, theta=0.5,
                                      load_scale=load_scale), sched)
    net.trace_edge = edge_key
    for t in range(HORIZON):
        net.step(t)
    rows = net.trace[::DECIMATE]
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    q = sum(r["servedQ"] for r in rows)
    p = sum(r["servedP"] for r in rows)
    print(f"  {name}: {len(rows)} rows, edge {edge_key[0]} -> {edge_key[1]}, "
          f"served {q/(q+p+1e-12):.3f} on Q-keys, "
          f"mean surrogate {sum(r['ageQ'] for r in rows)/len(rows):.1f} slots")


def busiest(load_scale: float, pilot: int = 3000):
    """Pick the edges worth tracing, by measured service rather than by name.

    Which edges carry traffic is a property of the topology and the
    flows, not something to guess: on this instance four of the five
    flows have a gateway in common and never touch the mesh at all, so
    naming an arbitrary mesh edge traces an idle link.
    """
    nodes, edges = A.default_topology()
    flows = A.scaled_flows()
    sched, _ = A.build_qkd_schedule(weather_seed=0, hours=12)
    net = A.HybridNetwork(nodes, edges, flows,
                          A.TqdConfig(theta=0.5, load_scale=load_scale),
                          sched)
    for t in range(pilot):
        net.step(t)
    gs = {g.name for g in A.DEFAULT_GROUND_STATIONS}
    tot = {k: net.cum_servedQ[k] + net.cum_servedP[k] for k in net.C}
    gs_edge = max((k for k in tot if k in net.q_arcs), key=lambda k: tot[k])
    mesh = [k for k in tot if k not in net.q_arcs]
    mesh_edge = max(mesh, key=lambda k: tot[k])
    print(f"  busiest quantum-capable edge {gs_edge[0]} -> {gs_edge[1]}, "
          f"{tot[gs_edge]:.0f} blocks in the pilot")
    print(f"  busiest mesh edge {mesh_edge[0]} -> {mesh_edge[1]}, "
          f"{tot[mesh_edge]:.0f} blocks, all of them on manufactured keys")
    return gs_edge, mesh_edge


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    load = 3.0
    gs_edge, mesh_edge = busiest(load)
    trace_one(gs_edge, "trace_gs.csv", load_scale=load)
    trace_one(mesh_edge, "trace_mesh.csv", load_scale=load)


if __name__ == "__main__":
    main()
