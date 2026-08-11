"""
Vector-PDF figure generation for Paper 2 (Age-of-Secret Routing).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
SIM  = HERE.parent
RESULTS = SIM / "results"
FIGS = SIM / "figs"
FIGS.mkdir(exist_ok=True, parents=True)
PAPER = SIM.parent

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

SCHED_ORDER = ["shortest_path", "pqc_only", "qkd_only",
               "key_rate_aware", "aos_ideal", "aos_backpressure",
               "aos_greedy", "aos_cg"]
SCHED_LABEL = {
    "shortest_path":     "Shortest path",
    "pqc_only":          "PQC-only",
    "qkd_only":          "QKD-only",
    "key_rate_aware":    "Key-rate-aware",
    "aos_ideal":         "AoS-BP-Ideal",
    "aos_backpressure":  "AoS-BP-H",
    "aos_greedy":        "AoS-BP-G",
    "aos_cg":            "AoS-BP",
}
# Categorical palette, fixed order, never cycled.  Validated with the
# dataviz six-checks validator: lightness band PASS, chroma floor PASS,
# adjacent-pair CVD separation PASS (worst 9.6 protan), normal-vision
# floor PASS (worst 15.6).  Contrast-vs-surface WARNs on the three
# lightest hues; the relief is Table I, which carries every value
# numerically, plus direct labels on the trajectory figures.
# The main algorithm (aos_cg) takes the strongest hue; the greedy
# ablation (aos_greedy) sits next to it in an olive that is separable
# from the blue under all three CVD simulations.
SCHED_FACE = {
    "shortest_path":     "#56B4E9",
    "pqc_only":          "#009E73",
    "qkd_only":          "#E69F00",
    "key_rate_aware":    "#D55E00",
    "aos_ideal":         "#7B5AA6",
    "aos_backpressure":  "#CC79A7",
    "aos_greedy":        "#8C8C00",
    "aos_cg":            "#0072B2",
}
SCHED_EDGE = {
    "shortest_path":     "#22526a",
    "pqc_only":          "#005c3f",
    "qkd_only":          "#8a5e00",
    "key_rate_aware":    "#7a3500",
    "aos_ideal":         "#4a3566",
    "aos_backpressure":  "#7a3f68",
    "aos_greedy":        "#4e4e00",
    "aos_cg":            "#003255",
}
SCEN_ORDER = ["nominal", "weather", "relay_compromise",
              "traffic_surge", "coalition_partition"]
SCEN_LABEL = {
    "nominal":             "Nominal",
    "weather":             "Weather",
    "relay_compromise":    "Relay",
    "traffic_surge":       "Surge",
    "coalition_partition": "Partition",
}


def load_master() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "master.csv")
    df["scheduler"] = pd.Categorical(df["scheduler"], categories=SCHED_ORDER)
    df["scenario"]  = pd.Categorical(df["scenario"],  categories=SCEN_ORDER)
    return df


# ---------------------------------------------------------------------------
# Figure 1: Topology / architecture (TikZ-style native python diagram)
# ---------------------------------------------------------------------------

def fig_topology(out_name: str = "fig_aos_topology"):
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.set_xlim(0, 14.4); ax.set_ylim(0, 5.0); ax.set_axis_off()

    def box(x, y, w, h, txt, face, edge, fontsize=8.5):
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.06,rounding_size=0.16",
                           ec=edge, fc=face, lw=1.0)
        ax.add_patch(p)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center",
                fontsize=fontsize)

    # LEO constellation cloud
    leo = FancyBboxPatch((4.2, 3.6), 6.0, 1.2,
                         boxstyle="round,pad=0.08,rounding_size=0.25",
                         ec="#888888", fc="#f0f0f0", lw=1.0, ls="--")
    ax.add_patch(leo)
    ax.text(7.2, 4.2, "Hypothetical QKD-capable LEO layer: 1306 real Starlink\nPhase-1 TLEs, 52.5-53.5° incl., 530-570 km (May 2026)",
            ha="center", va="center", fontsize=8.5, style="italic",
            color="#555555")

    # Ground stations
    gs = [("Waco-TX", 0.4),  ("FortBragg-NC", 2.5),
          ("Ramstein", 4.7), ("Yokota", 7.0),
          ("DiegoGarcia", 9.4), ("CampLemonnier", 11.7)]
    for name, x in gs:
        box(x, 0.4, 2.1, 0.9, name, "#E8EEF3", "#5C7080", fontsize=8)

    # Gateway mesh row
    gw = [("GW-NAWest", 1.4), ("GW-NAEast", 4.5),
          ("GW-EU",     8.0), ("GW-APAC",   11.0)]
    for name, x in gw:
        box(x, 2.0, 2.1, 0.9, name, "#FFE6A8", "#A8771C", fontsize=8.5)

    # QKD links (dashed) from LEO cloud to GS
    for _, x in gs:
        ax.annotate("", xy=(x + 1.05, 1.3),
                    xytext=(x + 1.05, 3.6),
                    arrowprops=dict(arrowstyle="-", lw=0.5, ls=":",
                                    color="#888888"))
    # Classical links: GS to its primary GW (solid)
    gs_to_gw = [(0.4, 1.4), (2.5, 4.5), (4.7, 8.0),
                (7.0, 11.0), (9.4, 11.0), (11.7, 8.0)]
    for x_gs, x_gw in gs_to_gw:
        ax.annotate("", xy=(x_gw + 1.05, 2.0),
                    xytext=(x_gs + 1.05, 1.3),
                    arrowprops=dict(arrowstyle="-", lw=0.7, color="#1F4E79"))

    # GW mesh links (curved, faint)
    gw_x = [1.4 + 1.05, 4.5 + 1.05, 8.0 + 1.05, 11.0 + 1.05]
    for i in range(len(gw_x)):
        for j in range(i + 1, len(gw_x)):
            ax.annotate("", xy=(gw_x[j], 2.5), xytext=(gw_x[i], 2.5),
                        arrowprops=dict(arrowstyle="-", lw=0.5,
                                        color="#1F4E79", alpha=0.4))

    # Caption-side legend
    ax.text(0.1, 4.5, "···  QKD overlay (opportunistic key supply)",
            fontsize=8, color="#555555")
    ax.text(0.1, 4.15, "─── Classical data plane (PQC-protected)",
            fontsize=8, color="#1F4E79")

    fig.savefig(FIGS / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{out_name}.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Secure-goodput bar chart (5 schedulers x 5 scenarios)
# ---------------------------------------------------------------------------

def fig_goodput(df: pd.DataFrame, out_name: str = "fig_aos_goodput"):
    agg = df.groupby(["scenario", "scheduler"], observed=True)["secure_goodput_bps"]\
            .agg(["mean", "std"]).reset_index()
    # Designed at IEEE single-column width (3.4 in) so width=\linewidth
    # renders at 1:1 with no font shrinkage.
    with plt.rc_context({
        "font.size": 10, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9,
    }):
        fig, ax = plt.subplots(figsize=(3.6, 2.9))
        x = np.arange(len(SCEN_ORDER))
        width = 0.13
        centre = (len(SCHED_ORDER) - 1) / 2.0
        for i, sc in enumerate(SCHED_ORDER):
            s = agg[agg["scheduler"] == sc].set_index("scenario").reindex(SCEN_ORDER)
            ax.bar(x + (i - centre) * width, s["mean"].values / 1e6, width,
                   yerr=s["std"].values / 1e6, capsize=2,
                   label=SCHED_LABEL[sc], facecolor=SCHED_FACE[sc],
                   edgecolor=SCHED_EDGE[sc], linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([SCEN_LABEL[s] for s in SCEN_ORDER],
                           rotation=20, ha="right",
                           rotation_mode="anchor")
        ax.set_ylabel("Secure goodput (Mbps)", labelpad=2)
        ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.30),
                  frameon=False, handlelength=1.4, columnspacing=1.2,
                  handletextpad=0.4)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        # Explicit subplots_adjust avoids the bbox_inches='tight' bug that
        # truncates the top of rotated y-axis labels.  Manually leave room
        # for the rotated x-tick labels (bottom) and the external legend.
        # Wider bottom margin for the 3-row, 2-column legend.
        fig.subplots_adjust(left=0.16, right=0.97, top=0.96, bottom=0.46)
        fig.savefig(FIGS / f"{out_name}.pdf")
        fig.savefig(FIGS / f"{out_name}.png")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: AoS mean bar chart
# ---------------------------------------------------------------------------

def fig_aos(df: pd.DataFrame, out_name: str = "fig_aos_mean"):
    """Mean Age of Secret per scenario, as a dot plot on a log axis.

    Deliberately not a grouped bar chart.  The values span two decades
    (AoS-BP near 2 s against AoS-BP-Ideal near 180 s), so on a linear bar
    axis the result the paper is about collapses into an invisible sliver
    against the baselines.  Log-scaled *bars* are not the fix either:
    a bar encodes magnitude as length from zero, which a log axis breaks.
    A dot carries no area, so it scales honestly.
    """
    agg = (df.groupby(["scenario", "scheduler"], observed=True)["mean_aos"]
             .mean().reset_index())
    with plt.rc_context({"font.size": 9, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 7.5}):
        fig, ax = plt.subplots(figsize=(3.4, 2.9))
        ypos = {s: i for i, s in enumerate(reversed(SCEN_ORDER))}
        for scen, y in ypos.items():
            vals = [agg[(agg.scenario == scen)
                        & (agg.scheduler == sc)]["mean_aos"].values
                    for sc in SCHED_ORDER]
            vals = [v[0] for v in vals if len(v)]
            if vals:
                # connector line makes the spread per scenario legible
                ax.plot([min(vals), max(vals)], [y, y], color="#cccccc",
                        lw=1.0, zorder=0, solid_capstyle="round")
        for sc in SCHED_ORDER:
            d = agg[agg.scheduler == sc]
            xs = [d[d.scenario == s]["mean_aos"].values for s in ypos]
            xs = [(v[0] if len(v) else np.nan) for v in xs]
            ax.scatter(xs, list(ypos.values()), s=42,
                       facecolor=SCHED_FACE[sc], edgecolor=SCHED_EDGE[sc],
                       linewidth=0.7, label=SCHED_LABEL[sc], zorder=3,
                       clip_on=False)
        ax.set_xscale("log")
        lo, hi = agg["mean_aos"].min(), agg["mean_aos"].max()
        ax.set_xlim(lo / 1.6, hi * 1.6)   # keep end markers off the spines
        ax.set_yticks(list(ypos.values()))
        ax.set_yticklabels([SCEN_LABEL[s] for s in ypos])
        ax.set_xlabel("Mean Age of Secret (s), log scale")
        ax.set_ylim(-0.6, len(ypos) - 0.4)
        ax.grid(axis="x", which="major", linestyle=":", linewidth=0.5,
                alpha=0.6)
        ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.45, -0.28),
                  frameon=False, handlelength=1.0, columnspacing=1.0,
                  handletextpad=0.3, scatterpoints=1)
        fig.subplots_adjust(left=0.22, right=0.98, top=0.97, bottom=0.42)
        fig.savefig(FIGS / f"{out_name}.pdf")
        fig.savefig(FIGS / f"{out_name}.png")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Goodput vs AoS Pareto
# ---------------------------------------------------------------------------

def fig_pareto(df: pd.DataFrame, out_name: str = "fig_aos_pareto"):
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for sc in SCHED_ORDER:
        s = df[df["scheduler"] == sc]
        gp = s["secure_goodput_bps"].mean() / 1e6
        aos = s["mean_aos"].mean()
        ax.scatter(aos, gp, s=80, facecolor=SCHED_FACE[sc],
                   edgecolor=SCHED_EDGE[sc], linewidth=0.9,
                   label=SCHED_LABEL[sc])
    ax.set_xlabel("Mean Age of Secret (s)")
    ax.set_ylabel("Secure goodput (Mbps)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIGS / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{out_name}.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Lyapunov drift verification (queue + key-deficit trajectories)
# ---------------------------------------------------------------------------

def fig_lyapunov(out_name: str = "fig_lyapunov"):
    """Plot the joint Lyapunov L(X_t) = 0.5||Q||^2 + 0.5 omega ||Z||^2
    trajectory under three schedulers, on a log-y axis.  Shows that
    shortest-path diverges while both AoS-aware schedulers stay
    bounded as Theorem 1 predicts."""
    # Use the extended 3600-cycle run if available (results_lyap),
    # otherwise fall back to the main 600-cycle results.
    per_step_dir = (SIM / "results_lyap") if (SIM / "results_lyap").exists() \
                  else RESULTS
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    # Colour follows the entity: same scheduler, same hue as every other
    # figure.  Line style is the secondary encoding so identity never
    # rests on colour alone.
    panels = [
        ("shortest_path",    SCHED_LABEL["shortest_path"],    ":"),
        ("aos_ideal",        SCHED_LABEL["aos_ideal"],        "--"),
        ("aos_backpressure", SCHED_LABEL["aos_backpressure"], "-."),
        ("aos_greedy",       SCHED_LABEL["aos_greedy"],       "--"),
        ("aos_cg",           SCHED_LABEL["aos_cg"],           "-"),
    ]
    win = 200                                # ~3-min rolling average
    for sc, lab, ls in panels:
        try:
            df = pd.read_csv(per_step_dir / f"{sc}_nominal_s0.csv")
        except FileNotFoundError:
            continue
        # L(X_t) is logged directly by the simulator as
        #   0.5 * sum_{i,f} (Q^f_i)^2 + (omega/2) * sum_{ij} Z_ij^2
        # with Z the Eq. (5) virtual key-queue.  Earlier revisions of this
        # figure reconstructed a proxy here from (sum Q)^2 and the
        # memoryless deficit max(0, K_min - k_min); both were wrong, and the
        # proxy Z was bounded by construction so it could never diverge.
        L = df["lyapunov"].astype(np.float64).values + 1.0
        # Rolling mean smooths the transient queue/key oscillations so the
        # bounded vs. unbounded distinction is visually unambiguous.
        L_smooth = pd.Series(L).rolling(window=win, min_periods=1).mean().values
        ax.semilogy(df["t"], L_smooth, color=SCHED_FACE[sc], linestyle=ls,
                    label=lab, lw=1.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Joint Lyapunov $L(X_t)$")
    ax.set_ylim(1e-1, 1e24)
    ax.set_yticks([1e0, 1e4, 1e8, 1e12, 1e16, 1e20])
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    ax.grid(which="major", linestyle=":", linewidth=0.5, alpha=0.6)
    # Annotate the divergent vs. bounded regions.
    ax.text(0.04, 0.96, "unstable", transform=ax.transAxes, ha="left",
            va="top", fontsize=8, color=SCHED_EDGE["shortest_path"],
            fontweight="bold")
    ax.text(0.04, 0.04, "bounded", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=8, color=SCHED_EDGE["aos_cg"],
            fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{out_name}.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: K_max invariance of the routing weight (parameter study)
# ---------------------------------------------------------------------------

VARIANT_LABEL = {"W0": "$\\alpha(K^{\\max}\\!-\\!K)$ (conference)",
                 "W1": "$\\alpha(1\\!-\\!K/K^{\\max})$",
                 "W3": "demand-normalised",
                 "drift": "$\\omega\\rho Z$ (AoS-BP)"}
VARIANT_COLOR = {"W0": "#D55E00", "W1": "#E69F00",
                 "W3": "#009E73", "drift": "#0072B2"}
VARIANT_STYLE = {"W0": ":", "W1": "--", "W3": "-.", "drift": "-"}


def fig_kmax(out_name: str = "fig_kmax_invariance"):
    """Mean AoS against the key-buffer cap, per form of the key-scarcity
    term.  A horizontal line is an edge cost whose behaviour does not
    depend on an arbitrary buffer size; the conference form is not one."""
    src = SIM / "results_param" / "param_study.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    rows = []
    for v in ("W0", "W1", "W3"):
        d = df[(df.study == "A") & (df.horizon_s == 3600)
               & (df.weight_variant == v)].sort_values("k_max_bits")
        if len(d):
            rows.append((v, d))
    d = df[(df.study == "E") & (df.horizon_s == 3600)].sort_values("k_max_bits")
    if len(d):
        rows.append(("drift", d))
    ends = []
    for v, d in rows:
        ax.loglog(d.k_max_bits.values, d.mean_aos.values,
                  color=VARIANT_COLOR[v], linestyle=VARIANT_STYLE[v],
                  marker="o", markersize=4, lw=1.8, label=VARIANT_LABEL[v])
        ends.append([np.log10(d.mean_aos.values[-1]),
                     d.k_max_bits.values[-1],
                     "AoS-BP" if v == "drift" else v, VARIANT_COLOR[v]])
    # Direct labels so identity is never colour-alone.  Series that
    # converge would overprint, so push the labels apart in log space
    # before drawing them.
    ends.sort()
    min_gap = 0.13
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < min_gap:
            ends[i][0] = ends[i - 1][0] + min_gap
    for ly, x, lab, col in ends:
        ax.annotate(lab, xy=(x, 10 ** ly), xytext=(4, 0),
                    textcoords="offset points", fontsize=7, color=col,
                    va="center")
    ax.set_xlabel("Key-buffer cap $K^{\\max}$ (bits)")
    ax.set_ylabel("Mean Age of Secret (s)")
    ax.grid(which="major", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{out_name}.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: load sweep locating the empirical boundary of Lambda_S
# ---------------------------------------------------------------------------

def fig_loadsweep(out_name: str = "fig_load_sweep"):
    """Backlog growth rate against offered load.  Zero slope is stability;
    the load at which a scheduler lifts off the axis is its empirical
    capacity boundary."""
    src = SIM / "results_param" / "param_study.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    d = df[df.study == "D"]
    if not len(d):
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    # AoS-BP-H once produced slopes identical to AoS-BP and was drawn as a
    # wide halo beneath it.  That is no longer true above the stability
    # boundary, so it gets an ordinary line; a halo would now hide a real
    # difference rather than reveal a coincidence.
    style = {"shortest_path":    ("--", 1.8, 2),
             "aos_ideal":        ("--", 1.8, 2),
             "aos_backpressure": ("-.", 1.8, 2),
             "aos_greedy":       ("--", 1.4, 2),
             "aos_cg":           ("-",  1.8, 3)}
    order = ("aos_backpressure", "aos_greedy", "shortest_path",
             "aos_ideal", "aos_cg")
    for sc in order:
        g = d[d.scheduler == sc].sort_values("load_scale")
        if not len(g):
            continue
        ls, lw, z = style[sc]
        ax.plot(g.load_scale.values, g.queue_slope_bits_per_cycle.values,
                color=SCHED_FACE[sc], marker="o",
                markersize=4,
                lw=lw, label=SCHED_LABEL[sc], linestyle=ls, zorder=z,
                solid_capstyle="round")
    ax.set_yscale("symlog", linthresh=1e5)
    ax.axhline(0, color="#666666", lw=0.8)
    ax.set_xlabel("Offered load ($\\times$ nominal)")
    ax.set_ylabel("Backlog growth (bits/slot)")
    ax.grid(which="major", linestyle=":", linewidth=0.5, alpha=0.6)
    # AoS-BP-H is drawn first (as the halo) but should not lead the legend.
    h, lb = ax.get_legend_handles_labels()
    want = [SCHED_LABEL[k] for k in ("shortest_path", "aos_ideal",
                                     "aos_backpressure", "aos_greedy",
                                     "aos_cg")]
    idx = [lb.index(w) for w in want if w in lb]
    ax.legend([h[i] for i in idx], [lb[i] for i in idx],
              loc="upper left", frameon=False, fontsize=7)
    # Sits over the flat (zero-slope) stretch on the left, not the
    # divergent right-hand side.
    ax.text(0.10, 0.10, "stable", transform=ax.transAxes, fontsize=8,
            color="#0072B2", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{out_name}.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1(b,c): the key-supply mechanism on a single edge
# ---------------------------------------------------------------------------

def fig_mechanism(out_name: str = "fig_mechanism", horizon: int = 600):
    """Trace the key-supply mechanism through a nominal run.

    Three stacked axes sharing time, never one axis with two scales.

    (i)  Supply rate on a busy QKD-capable edge.  QKD arrives from the
         pass schedule and varies continuously with elevation and
         weather; PQC is a flat capability underneath it.  This is the
         picture of key supply as a stochastic exogenous resource.
    (ii) Key pools.  The busy edge is drawn down by traffic and
         replenished; a lightly used gateway-mesh edge sits flat at its
         demand-sized reserve because the gate has closed.  Same units,
         so one axis, and the contrast is where the saving comes from.
    (iii) The resulting Age of Secret on the busy edge.

    Trace cached under sim/results_trace/ so the figure regenerates from
    a committed artifact.
    """
    import aos_network as A

    trace_dir = SIM / "results_trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    cache = trace_dir / "edge_trace.csv"

    if not cache.exists():
        sched, _ = A.build_qkd_schedule(weather_seed=0, hours=12)
        cfg = A.SimConfig(horizon_s=horizon, seed=0,
                          scheduler="aos_cg", scenario="nominal")
        nodes, edges = A.default_topology()
        net = A.AoSNetwork(nodes, edges, A.default_flows(), cfg, sched)
        gs_names = {g.name for g in A.DEFAULT_GROUND_STATIONS}
        rows = []
        for t in range(horizon):
            net.step(t)
            for e in net.edges:            # every edge, mesh included
                ek = e.key()
                gs = (e.src if e.src in gs_names
                      else (e.dst if e.dst in gs_names else None))
                rows.append(dict(
                    t=t, edge=f"{ek[0]}->{ek[1]}", qkd_capable=e.qkd_capable,
                    k=net.K[ek], aos=net.edge_aos(e, t),
                    served=net._D_cycle[ek], pqc=net._gen_cycle[ek],
                    qkd=(A.qkd_rate_at(t, sched.get(gs, [])) if gs else 0.0),
                    target=min(e.k_max_bits,
                               max(e.k_min_bits,
                                   cfg.pqc_reserve_s * net.u_hat[ek]))))
        pd.DataFrame(rows).to_csv(cache, index=False)

    df = pd.read_csv(cache)
    qk = df.groupby("edge")["qkd_capable"].first()
    busy = df.groupby("edge")["served"].sum()[qk].idxmax()
    d = df[df.edge == busy].sort_values("t")
    tv = d["t"].values
    rho = 1.0 / 256.0

    # QKD is deposited first and ungated, and the pool is far from its cap,
    # so QKD generation equals the pass rate and the remainder of the
    # generation total is the gated PQC contribution.
    W = 15          # rolling window; Lambda_S constrains time averages,
                    # and per-slot service is an on/off decision whose
                    # spikes would swamp the supply curves.
    sm = lambda v: pd.Series(v).rolling(W, min_periods=1, center=True).mean().values
    qkd = sm(d["qkd"].values)
    pqc = sm(np.maximum(0.0, d["pqc"].values - d["qkd"].values))
    demand = sm(rho * d["served"].values)

    supply, pqcc, dem = "#E69F00", "#0072B2", "#CC79A7"
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 7}):
        fig, ax = plt.subplots(2, 1, figsize=(7.0, 2.35), sharex=True,
                               gridspec_kw={"height_ratios": [1.5, 1.0],
                                            "hspace": 0.16})
        # (i) the resource balance, all three series in the same units
        # Supply as a part-to-whole stack, demand as the line riding on
        # it: the visual question is whether the dashed line stays inside
        # the filled envelope, which is exactly the Lambda_S constraint.
        ax[0].stackplot(tv, pqc / 1e3, qkd / 1e3,
                        colors=[pqcc, supply], alpha=0.9,
                        edgecolor="white", linewidth=0.7,
                        labels=["PQC supply (gated)",
                                "QKD supply (passes)"])
        ax[0].plot(tv, demand / 1e3, color=dem, lw=1.5, ls="--",
                   label=r"key demand $\rho D_e$")
        ax[0].set_ylabel("Key rate\n(kb/s)")
        ax[0].set_ylim(0, 1.6 * max(demand.max(), (qkd + pqc).max()) / 1e3)
        ax[0].legend(loc="upper center", frameon=False, ncol=3,
                     handlelength=1.4, columnspacing=1.0,
                     borderaxespad=0.1)
        # (ii) the metric that results
        ax[1].plot(tv, d["aos"].values, color=pqcc, lw=1.4)
        ax[1].set_ylabel("$\\mathrm{AoS}_e$\n(s)")
        ax[1].set_xlabel("Time (s)")
        for a_ in ax:
            a_.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        fig.subplots_adjust(left=0.085, right=0.995, top=0.98, bottom=0.17)
        fig.savefig(FIGS / f"{out_name}.pdf")
        fig.savefig(FIGS / f"{out_name}.png", dpi=200)
        plt.close(fig)
    return busy


# ---------------------------------------------------------------------------
# Data export for the pgfplots figures in the journal manuscript
# ---------------------------------------------------------------------------

def export_pgf_data():
    """Write the plotted series as plain tables for pgfplots.

    The journal manuscript draws its figures natively in pgfplots so that
    figure text is typeset by LaTeX in the document font, rather than
    embedded in a PDF in matplotlib's sans face.  The numbers are the same
    ones the matplotlib figures use; only the renderer differs.  Long
    traces are decimated, which is lossless here because they are already
    rolling-averaged.
    """
    out = PAPER / "journal" / "data"
    if not (PAPER / "journal").is_dir():
        return
    out.mkdir(parents=True, exist_ok=True)
    df = load_master()

    # (1) mean AoS per scenario per scheduler, and (2) the Pareto point
    agg = (df.groupby(["scenario", "scheduler"], observed=True)
             .agg(aos=("mean_aos", "mean"),
                  gp=("secure_goodput_bps", "mean")).reset_index())
    with open(out / "aos_by_scenario.dat", "w") as fh, \
         open(out / "aos_connector.dat", "w") as cf:
        fh.write("y scenario " + " ".join(SCHED_ORDER) + "\n")
        cf.write("x y\n")
        for y, sc in enumerate(reversed(SCEN_ORDER)):
            vals = [agg[(agg.scenario == sc) & (agg.scheduler == k)]
                    ["aos"].values[0] for k in SCHED_ORDER]
            fh.write(f"{y} {SCEN_LABEL[sc]} "
                     + " ".join(f"{v:.4f}" for v in vals) + "\n")
            # blank-line-separated segments; pgfplots breaks the path on
            # an empty line, giving one connector per scenario row
            cf.write(f"{min(vals):.4f} {y}\n{max(vals):.4f} {y}\n\n")
    o = df.groupby("scheduler", observed=True).agg(
        aos=("mean_aos", "mean"), gp=("secure_goodput_bps", "mean"))
    # Emitted as ready-made \\addplot lines rather than a table: one mark
    # per scheduler needs one plot per row, and row selection inside a
    # shared table is a pgfplots-version minefield.
    tikzcol = {"shortest_path": "cShort", "pqc_only": "cPQC",
               "qkd_only": "cQKD", "key_rate_aware": "cKey",
               "aos_ideal": "cIdeal", "aos_backpressure": "cBPH",
               "aos_greedy": "cBPG", "aos_cg": "cBP"}
    with open(out / "pareto_plots.tex", "w") as fh:
        fh.write("% generated by make_figures.py; do not edit\n")
        for k in SCHED_ORDER:
            if k not in o.index:
                continue
            big = "2.6pt" if k == "aos_cg" else "2.1pt"
            fh.write(
                f"\\addplot[only marks, mark=*, mark size={big}, "
                f"draw=black!50, fill={tikzcol[k]}, line width=.4pt] "
                f"coordinates {{({o.loc[k,'aos']:.4f},"
                f"{o.loc[k,'gp']/1e6:.3f})}};\n"
                f"\\addlegendentry{{{SCHED_LABEL[k]}}}\n")

    ps = SIM / "results_param" / "param_study.csv"
    if ps.exists():
        d = pd.read_csv(ps)
        # (3) K_max invariance, 3600-slot horizon
        with open(out / "kmax.dat", "w") as fh:
            fh.write("kmax W0 W1 W3 drift\n")
            a = d[(d.study == "A") & (d.horizon_s == 3600)]
            e = d[(d.study == "E") & (d.horizon_s == 3600)]
            for km in sorted(a.k_max_bits.unique()):
                row = [a[(a.k_max_bits == km) & (a.weight_variant == v)]
                       ["mean_aos"] for v in ("W0", "W1", "W3")]
                dr = e[e.k_max_bits == km]["mean_aos"]
                fh.write(f"{km:.0f} "
                         + " ".join(f"{r.values[0]:.4f}" for r in row)
                         + f" {dr.values[0]:.4f}\n")
        # (4) load sweep
        ld = d[d.study == "D"]
        with open(out / "loadsweep.dat", "w") as fh:
            cols = ["shortest_path", "aos_ideal", "aos_backpressure",
                    "aos_greedy", "aos_cg"]
            fh.write("load " + " ".join(cols) + "\n")
            for ls in sorted(ld.load_scale.unique()):
                r = [ld[(ld.load_scale == ls) & (ld.scheduler == c)]
                     ["queue_slope_bits_per_cycle"] for c in cols]
                fh.write(f"{ls} " + " ".join(
                    f"{max(0.0, x.values[0]):.6g}" if len(x) else "nan"
                    for x in r) + "\n")

    # (4b) utility-delay curve in chi (the drift-plus-penalty V knob)
    if ps.exists():
        d = pd.read_csv(ps)
        f = d[d.study == "F"].sort_values("chi_prime")
        if len(f):
            with open(out / "tradeoff.dat", "w") as fh:
                fh.write("chi aos queue\n")
                for _, r in f.iterrows():
                    fh.write(f"{r.chi_prime:.6g} {r.mean_aos:.4f} "
                             f"{max(r.mean_queue_bits, 0.0):.6g}\n")

    # (4c) what the exact per-slot solve costs and what it buys
    if ps.exists():
        d = pd.read_csv(ps)
        g = d[d.study == "G"].sort_values("load_scale")
        if len(g):
            with open(out / "cggap.dat", "w") as fh:
                fh.write("load rounds gap gapmax cg_aos gd_aos "
                         "cg_gp gd_gp\n")
                for _, r in g.iterrows():
                    fh.write(f"{r.load_scale:.6g} {r.cg_mean_iters:.4f} "
                             f"{100*r.cg_gap_frac:.4f} "
                             f"{100*r.cg_gap_max:.4f} "
                             f"{r.mean_aos:.4f} {r.greedy_aos:.4f} "
                             f"{r.goodput_mbps:.4f} "
                             f"{r.greedy_goodput_mbps:.4f}\n")

    # (5) Lyapunov trajectories, decimated
    lyd = SIM / "results_lyap"
    scheds = ["shortest_path", "aos_ideal", "aos_backpressure",
              "aos_greedy", "aos_cg"]
    series, tcol = {}, None
    for sc in scheds:
        f = lyd / f"{sc}_nominal_s0.csv"
        if not f.exists():
            continue
        g = pd.read_csv(f)
        L = pd.Series(g["lyapunov"].astype(float) + 1.0).rolling(
            200, min_periods=1).mean().values[::8]
        series[sc] = L
        tcol = g["t"].values[::8]
    if series:
        n = min(len(v) for v in series.values())
        with open(out / "lyapunov.dat", "w") as fh:
            fh.write("t " + " ".join(series) + "\n")
            for i in range(n):
                fh.write(f"{tcol[i]} " + " ".join(
                    f"{series[k][i]:.6g}" for k in series) + "\n")

    # (6) mechanism trace, decimated
    tf = SIM / "results_trace" / "edge_trace.csv"
    if tf.exists():
        t = pd.read_csv(tf)
        qk = t.groupby("edge")["qkd_capable"].first()
        busy = t.groupby("edge")["served"].sum()[qk].idxmax()
        b = t[t.edge == busy].sort_values("t")
        W, rho = 15, 1.0 / 256.0
        sm = lambda v: pd.Series(v).rolling(W, min_periods=1,
                                            center=True).mean().values
        qkd = sm(b["qkd"].values)
        pqc = sm(np.maximum(0.0, b["pqc"].values - b["qkd"].values))
        dem = sm(rho * b["served"].values)
        aos = b["aos"].values
        with open(out / "mechanism.dat", "w") as fh:
            fh.write("t pqc qkd total demand aos\n")
            for i in range(0, len(b), 2):
                fh.write(f"{b['t'].values[i]} {pqc[i]/1e3:.4f} "
                         f"{qkd[i]/1e3:.4f} {(pqc[i]+qkd[i])/1e3:.4f} "
                         f"{dem[i]/1e3:.4f} {aos[i]:.4f}\n")
    print("exported pgfplots data to", out)


def main():
    fig_topology()
    df = load_master()
    fig_goodput(df)
    fig_aos(df)
    fig_pareto(df)
    fig_lyapunov()
    fig_kmax()
    fig_loadsweep()
    print('mechanism edge:', fig_mechanism())
    export_pgf_data()
    # Copy figures into paper directory
    for name in ("fig_aos_topology", "fig_aos_goodput", "fig_aos_mean",
                 "fig_aos_pareto", "fig_lyapunov"):
        src = FIGS / f"{name}.pdf"
        if src.exists():
            (PAPER / f"{name}.pdf").write_bytes(src.read_bytes())
    # Journal build gets the same set plus the two parameter-study figures.
    journal = PAPER / "journal"
    if journal.is_dir():
        for name in ("fig_aos_goodput", "fig_aos_mean", "fig_aos_pareto",
                     "fig_lyapunov", "fig_kmax_invariance",
                     "fig_load_sweep", "fig_mechanism"):
            src = FIGS / f"{name}.pdf"
            if src.exists():
                (journal / f"{name}.pdf").write_bytes(src.read_bytes())
    print("Wrote figures to", FIGS)

    # Print headline table for the manuscript
    print("\nHeadline table (mean across seeds):")
    g = df.groupby(["scenario", "scheduler"], observed=True).agg(
        goodput=("secure_goodput_bps", "mean"),
        outage=("secrecy_outage_rate", "mean"),
        aos=("mean_aos", "mean"),
    ).round(3)
    g["goodput"] = (g["goodput"] / 1e6).round(2)
    print(g.to_string())


if __name__ == "__main__":
    main()
