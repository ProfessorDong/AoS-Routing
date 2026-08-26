"""How the per-slot cost of the policy grows with the network.

The evaluation instance is deliberately small, because its purpose is to
compare a computed capacity boundary against measured behavior and that
comparison needs an instance whose linear program can be solved exactly.
A separate question is whether the policy itself stays affordable, and
that is what this script measures.

Three costs are separated, because they scale differently.

  ROUTING is one Dijkstra per arriving class per slot on the transformed
  graph, so it is O(|C| |E| log|V|) and grows with both the network and
  the number of classes.

  MANUFACTURING is the packing program of Proposition 2(iii), one linear
  program per slot over the edges with positive weight.  It does not
  depend on the number of classes at all, only on the graph.

  STATE is the key banks.  Each holds at most T_a age-distinguished
  entries per edge per species, so memory is O(|E| T_a) and independent
  of load.

Topologies are generated parametrically: G ground stations, each
attached to two of W gateways, with the gateways fully meshed.  The pass
schedule of the six real stations is reused cyclically, since the point
here is timing rather than orbital fidelity.

Run:  python3 scaling.py
"""
from __future__ import annotations

import csv
import time
import tracemalloc
from pathlib import Path

import numpy as np

import aos_tqd as A
from aos_network import DEFAULT_GROUND_STATIONS, Edge, Flow

OUT = Path(__file__).resolve().parent.parent / "results_tqd"
SLOTS = 60           # timed slots per size, after a warm start
WARMUP = 30
SIZES = [(6, 4), (12, 6), (24, 10), (48, 16), (96, 24)]


def topology(n_gs: int, n_gw: int):
    gs = [f"GS{i:03d}" for i in range(n_gs)]
    gw = [f"GW{i:03d}" for i in range(n_gw)]
    edges = []
    for i in range(n_gw):
        for j in range(i + 1, n_gw):
            edges.append(Edge(gw[i], gw[j], capacity_bps=10e9, latency_ms=45.0))
            edges.append(Edge(gw[j], gw[i], capacity_bps=10e9, latency_ms=45.0))
    for i, g in enumerate(gs):
        for k in (0, 1):
            w = gw[(i + k) % n_gw]
            edges.append(Edge(g, w, capacity_bps=1e9, latency_ms=20.0 + 20 * k,
                              qkd_capable=True))
            edges.append(Edge(w, g, capacity_bps=1e9, latency_ms=20.0 + 20 * k,
                              qkd_capable=True))
    return gs + gw, edges, gs


def flows_for(gs: list[str], n_classes: int, total_mbps: float):
    per = total_mbps * 1e6 / n_classes
    out = []
    for i in range(n_classes):
        s, d = gs[i % len(gs)], gs[(i + len(gs) // 2 + 1) % len(gs)]
        if s == d:
            d = gs[(i + 1) % len(gs)]
        out.append(Flow(name=f"f{i}", src=s, dst=d, arrival_bps=per))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base, _ = A.build_qkd_schedule(weather_seed=0, hours=12)
    real = [g.name for g in DEFAULT_GROUND_STATIONS]
    rows = []
    print(f"{'nodes':>6s} {'edges':>6s} {'classes':>8s} {'ms/slot':>9s} "
          f"{'weights':>8s} {'mfg LP':>8s} {'MB':>7s}")
    for n_gs, n_gw in SIZES:
        nodes, edges, gs = topology(n_gs, n_gw)
        n_classes = 4 * n_gs
        flows = flows_for(gs, n_classes, 0.5 * n_gs)
        sched = {g: base[real[i % len(real)]] for i, g in enumerate(gs)}
        net = A.HybridNetwork(nodes, edges, flows,
                              A.TqdConfig(theta=0.5), sched)
        net.qkd_edges_of = {g: [e.key() for e in edges
                                if e.qkd_capable and g in (e.src, e.dst)]
                            for g in gs}
        for t in range(WARMUP):
            net.step(t)
        tracemalloc.start()
        t0 = time.perf_counter()
        for t in range(WARMUP, WARMUP + SLOTS):
            net.step(t)
        wall = (time.perf_counter() - t0) / SLOTS
        mb = tracemalloc.get_traced_memory()[1] / 2**20
        tracemalloc.stop()
        # attribute the two components this paper adds; the balance
        # is the per-class Dijkstra and the queue bookkeeping
        t0 = time.perf_counter()
        for _ in range(20):
            net._weights(WARMUP, set())
        w_ms = (time.perf_counter() - t0) / 20
        t0 = time.perf_counter()
        for _ in range(20):
            net._manufacture()
        m_ms = (time.perf_counter() - t0) / 20
        print(f"{len(nodes):6d} {len(edges):6d} {n_classes:8d} "
              f"{wall*1e3:9.2f} {w_ms*1e3:8.2f} {m_ms*1e3:8.2f} {mb:7.1f}")
        rows.append(dict(nodes=len(nodes), edges=len(edges),
                         classes=n_classes, ms_per_slot=wall * 1e3,
                         ms_weights=w_ms * 1e3, ms_manufacture=m_ms * 1e3,
                         peak_mb=mb))
    with open(OUT / "scaling.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote scaling.csv")
    r = rows[-1]
    print(f"\nAt {r['nodes']} nodes and {r['classes']} classes one slot "
          f"costs {r['ms_per_slot']:.1f} ms, so a one-second slot leaves "
          f"a margin of {1000/max(r['ms_per_slot'],1e-9):.0f}x.")


if __name__ == "__main__":
    main()
