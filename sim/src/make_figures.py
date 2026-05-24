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
               "key_rate_aware", "aos_ideal", "aos_backpressure"]
SCHED_LABEL = {
    "shortest_path":     "Shortest path",
    "pqc_only":          "PQC-only",
    "qkd_only":          "QKD-only",
    "key_rate_aware":    "Key-rate-aware",
    "aos_ideal":         "AoS-BP-Ideal",
    "aos_backpressure":  "AoS-BP",
}
SCHED_FACE = {
    "shortest_path":     "#56B4E9",
    "pqc_only":          "#009E73",
    "qkd_only":          "#E69F00",
    "key_rate_aware":    "#D55E00",
    "aos_ideal":         "#999999",
    "aos_backpressure":  "#0072B2",
}
SCHED_EDGE = {
    "shortest_path":     "#22526a",
    "pqc_only":          "#005c3f",
    "qkd_only":          "#8a5e00",
    "key_rate_aware":    "#7a3500",
    "aos_ideal":         "#444444",
    "aos_backpressure":  "#003255",
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
    ax.text(7.2, 4.2, "Walker constellation (484 LEO sats, 22 planes × 22, 53° incl., 550 km)",
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
    agg = df.groupby(["scenario", "scheduler"], observed=True)["mean_aos"]\
            .agg(["mean", "std"]).reset_index()
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
            ax.bar(x + (i - centre) * width, s["mean"].values, width,
                   yerr=s["std"].values, capsize=2,
                   label=SCHED_LABEL[sc], facecolor=SCHED_FACE[sc],
                   edgecolor=SCHED_EDGE[sc], linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([SCEN_LABEL[s] for s in SCEN_ORDER],
                           rotation=20, ha="right",
                           rotation_mode="anchor")
        ax.set_ylabel("Mean Age of Secret (s)")
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
    panels = [
        ("shortest_path",    "Shortest path",         "#D55E00", ":"),
        ("aos_ideal",        "AoS-BP-Ideal (Thm. 1)", "#666666", "--"),
        ("aos_backpressure", "AoS-BP (Alg. 1)",       "#0072B2", "-"),
    ]
    omega = 200.0
    win = 200                                # ~3-min rolling average
    for sc, lab, col, ls in panels:
        try:
            df = pd.read_csv(per_step_dir / f"{sc}_nominal_s0.csv")
        except FileNotFoundError:
            continue
        q     = df["q_total"].astype(np.float64).values
        k_min = df["k_min"].astype(np.float64).values
        z     = np.maximum(0.0, 50_000.0 - k_min)
        L     = 0.5 * (q ** 2) + 0.5 * omega * (z ** 2) + 1.0
        # Rolling mean smooths the transient queue/key oscillations so the
        # bounded vs. unbounded distinction is visually unambiguous.
        L_smooth = pd.Series(L).rolling(window=win, min_periods=1).mean().values
        ax.semilogy(df["t"], L_smooth, color=col, linestyle=ls,
                    label=lab, lw=1.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Joint Lyapunov $L(X_t)$")
    ax.set_ylim(1e-1, 1e24)
    ax.set_yticks([1e0, 1e4, 1e8, 1e12, 1e16, 1e20])
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    ax.grid(which="major", linestyle=":", linewidth=0.5, alpha=0.6)
    # Annotate the divergent vs. bounded regions.
    ax.text(0.04, 0.96, "unstable", transform=ax.transAxes, ha="left",
            va="top", fontsize=8, color="#D55E00", fontweight="bold")
    ax.text(0.04, 0.04, "bounded", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=8, color="#0072B2", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{out_name}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    fig_topology()
    df = load_master()
    fig_goodput(df)
    fig_aos(df)
    fig_pareto(df)
    fig_lyapunov()
    # Copy figures into paper directory
    for name in ("fig_aos_topology", "fig_aos_goodput", "fig_aos_mean",
                 "fig_aos_pareto", "fig_lyapunov"):
        src = FIGS / f"{name}.pdf"
        if src.exists():
            (PAPER / f"{name}.pdf").write_bytes(src.read_bytes())
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
