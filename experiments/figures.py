"""Figures.

  schedule_gap/  ring_validation, ring_depth, pilot_schedules, gap_location (light
                 and dark), fig13_ring_exact.png, fig14_gap_schedule.png, numbers.json
  q5_q1/         q1_depth, q1_collapse, limit_single_flip, limit_two_flip,
                 limit_collective, weighted_cliff, dehn_fit

Reads lr_p_sweep.csv, lr_p_sweep_eta.csv, gap_pilot.csv and q5_q1_numbers.json.

    python experiments/figures.py            # -> results/figures/
    python experiments/figures.py --only q5_q1
"""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullLocator

from maxcut_cudaq.exact import adiabatic_schedule, eta_infinite, ring_log2_success

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "figures"
PANELS = OUT / "schedule_gap"
Q51 = OUT / "q5_q1"

# Colors were checked for colorblind separation against each surface. Don't
# swap hues without checking again.
THEMES = {
    "light": dict(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        blue="#2a78d6",
        orange="#eb6834",
        aqua="#1baf7a",
        violet="#4a3aa7",
        ramp=["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
    ),
    "dark": dict(
        surface="#1a1a19",
        ink="#ffffff",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        blue="#3987e5",
        orange="#d95926",
        aqua="#199e70",
        violet="#9085e9",
        ramp=["#184f95", "#256abf", "#5598e7", "#9ec5f4"],
    ),
}
RING_DEPTHS = [5, 20, 100, 300]
HALF_SIZES = [16, 32, 64, 100, 128, 256]
SCHEDULES = [
    ("adiabatic", "gap-adapted"),
    ("linear", "linear ramp"),
    ("slow_ends", "slow at the ends"),
    ("slow_middle", "slow in the middle"),
]
PILOT_N = 14


# ----------------------------------------------------------------------------- data


def depth_for_half(n, adaptive):
    p = 2
    while ring_log2_success(n, p, u=adiabatic_schedule(n, p) if adaptive else None) < -1.0:
        p = max(p + 1, int(p * 1.05))
    return p


def collect():
    N = {}
    sweep = pd.read_csv(RESULTS / "lr_p_sweep.csv")
    sweep = sweep[sweep.family == "ring"]
    ns = np.arange(2, 42, 2)
    ring = {}
    for p in RING_DEPTHS:
        m = sweep[sweep.p == p].groupby("n_nodes").success_probability.mean()
        ring[p] = dict(
            n=ns.tolist(),
            exact=[ring_log2_success(int(n), p) for n in ns],
            eta=eta_infinite(p),
            sweep_n=m.index.tolist(),
            sweep=np.log2(m.values).tolist(),
        )
    N["ring"] = ring
    fit = pd.read_csv(RESULTS / "lr_p_sweep_eta.csv")
    fit = fit[fit.family == "ring"].set_index("p")
    N["eta"] = [
        dict(
            p=int(p),
            exact=eta_infinite(int(p)),
            fitted=float(fit.loc[p, "eta"]),
            C_fitted=float(fit.loc[p, "C"]),
        )
        for p in (5, 20, 60, 100, 200, 300)
    ]
    lin = [depth_for_half(n, False) for n in HALF_SIZES]
    ada = [depth_for_half(n, True) for n in HALF_SIZES]
    N["half"] = dict(
        n=HALF_SIZES,
        linear=lin,
        adapted=ada,
        slope_linear=float(np.polyfit(np.log(HALF_SIZES), np.log(lin), 1)[0]),
        slope_adapted=float(np.polyfit(np.log(HALF_SIZES), np.log(ada), 1)[0]),
    )

    d = pd.read_csv(RESULTS / "gap_pilot.csv")
    w = d.pivot_table(
        index=["n_nodes", "family", "seed", "p"], columns="schedule", values="success_probability"
    ).reset_index()
    at = w[w.n_nodes == PILOT_N]
    depths = sorted(at.p.unique())
    N["pilot"] = dict(
        n=PILOT_N,
        instances=int(at[at.p == depths[0]].shape[0]),
        p=[int(p) for p in depths],
        median={k: [float(at[at.p == p][k].median()) for p in depths] for k, _ in SCHEDULES},
        all_instances=int(w[w.p == depths[0]].shape[0]),
        beats_linear={
            k: [float((w[w.p == p][k] > w[w.p == p].linear).mean()) for p in depths]
            for k, _ in SCHEDULES
            if k != "linear"
        },
    )
    g = d.drop_duplicates(["n_nodes", "family", "seed"])
    loc = g.pivot_table(index="family", columns="n_nodes", values="u_at_gap_min", aggfunc="median")
    gap = g.pivot_table(index="family", columns="n_nodes", values="gap_min", aggfunc="median")
    N["gap"] = dict(
        n=[int(c) for c in loc.columns],
        location={f: [None if np.isnan(v) else float(v) for v in loc.loc[f]] for f in loc.index},
        size={f: [None if np.isnan(v) else float(v) for v in gap.loc[f]] for f in gap.index},
    )
    return N


# ----------------------------------------------------------------------------- style


def style(ax, T, title, subtitle, xlabel, ylabel):
    ax.set_facecolor(T["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(T["axis"])
        ax.spines[s].set_linewidth(1.0)
    ax.grid(True, color=T["grid"], lw=0.8, ls="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=T["muted"], labelsize=9.5, length=0, pad=5)
    ax.set_xlabel(xlabel, color=T["ink2"], fontsize=10, labelpad=7)
    ax.set_ylabel(ylabel, color=T["ink2"], fontsize=10, labelpad=7)
    ax.text(
        0,
        1.115,
        title,
        transform=ax.transAxes,
        color=T["ink"],
        fontsize=12.5,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax.text(
        0,
        1.045,
        subtitle,
        transform=ax.transAxes,
        color=T["ink2"],
        fontsize=9.5,
        va="bottom",
        ha="left",
    )


def end_label(ax, T, x, y, text, dy=0):
    ax.annotate(
        text,
        (x, y),
        xytext=(8, dy),
        textcoords="offset points",
        va="center",
        ha="left",
        color=T["ink2"],
        fontsize=9.5,
        annotation_clip=False,
    )


def legend(ax, T, handles, loc, **kw):
    leg = ax.legend(
        handles=handles,
        loc=loc,
        frameon=False,
        fontsize=9.3,
        handlelength=1.8,
        borderaxespad=0.6,
        labelspacing=0.45,
        **kw,
    )
    for t in leg.get_texts():
        t.set_color(T["ink2"])


def dots(T, color, size=6.5):
    return dict(marker="o", ms=size, mfc=color, mec=T["surface"], mew=1.6)


# ----------------------------------------------------------------------------- panels


def panel_ring_validation(ax, T, N):
    style(
        ax,
        T,
        "The exact ring formula reproduces the sweep",
        "linear ramp on the ring, four depths, Δγ = 0.6, Δβ = 0.3",
        "ring size n",
        "log₂ P(x*)",
    )
    for color, p in zip(T["ramp"], RING_DEPTHS):
        r = N["ring"][p]
        ax.plot([0, 40], [1, 1 - r["eta"] * 40], color=T["muted"], lw=0.9, zorder=2)
        ax.plot(r["n"], r["exact"], color=color, lw=2, solid_capstyle="round", zorder=3)
        ax.plot(r["sweep_n"], r["sweep"], ls="none", zorder=4, **dots(T, color))
        end_label(ax, T, 40, r["exact"][-1], f"p = {p}")
    ax.plot(
        [0],
        [1],
        ls="none",
        marker="o",
        ms=5,
        mfc=T["ink"],
        mec=T["surface"],
        mew=1.2,
        zorder=5,
        clip_on=False,
    )
    ax.annotate(
        "every asymptote passes\nthrough log₂ P = 1 at n = 0",
        (0, 1),
        xytext=(1.3, -7.4),
        textcoords="data",
        color=T["ink2"],
        fontsize=9.3,
        va="top",
        ha="left",
        arrowprops=dict(
            arrowstyle="-", color=T["muted"], lw=0.9, shrinkA=3, shrinkB=5, relpos=(0.0, 1.0)
        ),
    )
    ax.set_xlim(0, 40)
    ax.set_ylim(-16, 2)
    ax.set_xticks([0, 8, 16, 24, 32, 40])
    ax.set_yticks([0, -4, -8, -12, -16])
    legend(
        ax,
        T,
        [
            Line2D([], [], color=T["ramp"][2], lw=2, label="exact formula"),
            Line2D([], [], ls="none", label="CUDA-Q state vector (sweep)", **dots(T, T["ramp"][2])),
            Line2D([], [], color=T["muted"], lw=0.9, label="asymptote  −η(p)·n + 1"),
        ],
        "lower left",
    )


def panel_ring_depth(ax, T, N):
    h = N["half"]
    style(
        ax,
        T,
        "Layers needed for a 50% success rate on the ring",
        "exact formula; first depth that reaches P(x*) ≥ 0.5 on a 5%-spaced grid",
        "ring size n",
        "layers p",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    for key, color, name, slope in (
        ("linear", T["blue"], "linear ramp", h["slope_linear"]),
        ("adapted", T["orange"], "gap-adapted", h["slope_adapted"]),
    ):
        ax.plot(h["n"], h[key], color=color, lw=2, zorder=3, **dots(T, color))
        end_label(ax, T, h["n"][-1], h[key][-1], f"{name}\n$p \\propto n^{{{slope:.2f}}}$")
    i = h["n"].index(100)
    for key, off, ha in (("linear", (-9, 9), "right"), ("adapted", (9, -10), "left")):
        ax.annotate(
            f"{h[key][i]:,}",
            (100, h[key][i]),
            xytext=off,
            textcoords="offset points",
            ha=ha,
            va="center",
            color=T["ink"],
            fontsize=9.5,
            fontweight="bold",
        )
    ax.set_xlim(13, 300)
    ax.set_ylim(10, 40000)
    ax.set_xticks(h["n"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_yticks([10, 100, 1000, 10000])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    legend(
        ax,
        T,
        [
            Line2D(
                [], [], color=T["blue"], lw=2, label="linear ramp (LR-QAOA)", **dots(T, T["blue"])
            ),
            Line2D(
                [],
                [],
                color=T["orange"],
                lw=2,
                label="gap-adapted schedule",
                **dots(T, T["orange"]),
            ),
        ],
        "upper left",
    )


def panel_pilot_schedules(ax, T, N):
    P = N["pilot"]
    style(
        ax,
        T,
        "Slowing at the gap beats the linear ramp; blind shapes lose",
        f"median P(x*) over {P['instances']} instances, n = {P['n']}, 9 graph families",
        "layers p",
        "median P(x*)",
    )
    ax.set_xscale("log")
    look = {
        "adiabatic": (T["orange"], 2),
        "linear": (T["blue"], 2),
        "slow_ends": (T["muted"], 1.4),
        "slow_middle": (T["muted"], 1.4),
    }
    for key, name in SCHEDULES:
        color, lw = look[key]
        ax.plot(
            P["p"],
            P["median"][key],
            color=color,
            lw=lw,
            zorder=3 if lw == 2 else 2,
            **dots(T, color, 6.5 if lw == 2 else 5.5),
        )
        end_label(ax, T, P["p"][-1], P["median"][key][-1], name)
    ax.set_xlim(8.5, 118)
    ax.set_ylim(0, 1)
    ax.set_xticks(P["p"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    legend(
        ax,
        T,
        [
            Line2D(
                [],
                [],
                color=T["orange"],
                lw=2,
                label="gap-adapted (needs the spectrum)",
                **dots(T, T["orange"]),
            ),
            Line2D(
                [], [], color=T["blue"], lw=2, label="linear ramp (LR-QAOA)", **dots(T, T["blue"])
            ),
            Line2D(
                [],
                [],
                color=T["muted"],
                lw=1.4,
                label="blind non-linear shapes",
                **dots(T, T["muted"], 5.5),
            ),
        ],
        "upper left",
    )


def panel_gap_location(ax, T, N):
    G = N["gap"]
    style(
        ax,
        T,
        "Where the gap sits depends on the family",
        "layer fraction of the smallest relevant gap, median of 5 seeds",
        "graph size n",
        "fraction of the way through the layers",
    )
    ax.plot([7.6, 14], [1 / 3, 1 / 3], color=T["ink2"], lw=0.9, zorder=1)
    for fam, vals in G["location"].items():
        if fam in ("ring", "bipartite"):
            continue
        xy = [(n, v) for n, v in zip(G["n"], vals) if v is not None]
        ax.plot(
            *zip(*xy), color=T["muted"], lw=1.2, alpha=0.75, zorder=2, **dots(T, T["muted"], 4.8)
        )
    for fam, color, name in (
        ("ring", T["aqua"], "ring"),
        ("bipartite", T["violet"], "complete\nbipartite"),
    ):
        ax.plot(G["n"], G["location"][fam], color=color, lw=2, zorder=4, **dots(T, color))
        end_label(ax, T, G["n"][-1], G["location"][fam][-1], name)
    ax.set_xlim(7.6, 14.4)
    ax.set_ylim(0, 1)
    ax.set_xticks(G["n"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    legend(
        ax,
        T,
        [
            Line2D([], [], color=T["aqua"], lw=2, label="ring", **dots(T, T["aqua"])),
            Line2D(
                [], [], color=T["violet"], lw=2, label="complete bipartite", **dots(T, T["violet"])
            ),
            Line2D(
                [],
                [],
                color=T["muted"],
                lw=1.2,
                label="7 other families",
                **dots(T, T["muted"], 4.8),
            ),
            Line2D([], [], color=T["ink2"], lw=0.9, label="Δβ / (Δβ + Δγ) = 1/3"),
        ],
        "upper left",
        ncol=2,
        columnspacing=1.4,
        handletextpad=0.5,
    )


FIGURES = {
    "fig13_ring_exact": [
        ("ring_validation", panel_ring_validation),
        ("ring_depth", panel_ring_depth),
    ],
    "fig14_gap_schedule": [
        ("pilot_schedules", panel_pilot_schedules),
        ("gap_location", panel_gap_location),
    ],
}


def render(N):
    plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]
    plt.rcParams["mathtext.default"] = "regular"
    margins = dict(left=0.115, right=0.765, top=0.84, bottom=0.135)
    written = []
    for theme, T in THEMES.items():
        for name, panels in FIGURES.items():
            for key, draw in panels:
                fig, ax = plt.subplots(figsize=(6.4, 4.7), dpi=170)
                fig.patch.set_facecolor(T["surface"])
                fig.subplots_adjust(**margins)
                draw(ax, T, N)
                path = PANELS / f"{key}_{theme}.png"
                fig.savefig(path, facecolor=T["surface"])
                plt.close(fig)
                written.append(path)
            if theme == "light":
                fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), dpi=170)
                fig.patch.set_facecolor(T["surface"])
                fig.subplots_adjust(left=0.058, right=0.90, top=0.84, bottom=0.135, wspace=0.52)
                for ax, (_, draw) in zip(axes, panels):
                    draw(ax, T, N)
                path = OUT / f"{name}.png"
                fig.savefig(path, facecolor=T["surface"])
                plt.close(fig)
                written.append(path)
    return written


# ============================================================================ Q5 / Q1 figures
T = THEMES["light"]
NUM = RESULTS / "q5_q1_numbers.json"
RAMP3 = [T["ramp"][0], T["ramp"][1], T["ramp"][3]]
# linear ramp by bisection; the drawn line is the asymptote 0.266 n^2
LINEAR_N, LINEAR_P = [64, 128, 256], [1087, 4357, 17436]


def new_axes(width=6.4, height=4.7, **margins):
    fig, ax = plt.subplots(figsize=(width, height), dpi=170)
    fig.patch.set_facecolor(T["surface"])
    fig.subplots_adjust(**{**dict(left=0.115, right=0.765, top=0.84, bottom=0.135), **margins})
    return fig, ax


def integer_ticks(axis, ticks):
    axis.set_ticks(ticks)
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))


def q1_depth(R):
    D = R["q1_depths"]
    fig, ax = new_axes(right=0.66)
    style(
        ax,
        T,
        "Layers for a 50% success rate on the ring",
        "exact formula, Δγ = 0.6, Δβ = 0.3; first depth reaching P(x*) ≥ 0.5",
        "ring size n",
        "layers p",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    n = np.array([64, 2048])
    ax.plot(n, 0.266 * n**2, color=T["blue"], lw=2, zorder=3)
    ax.plot(LINEAR_N, LINEAR_P, ls="none", zorder=4, **dots(T, T["blue"]))
    end_label(ax, T, 2048, 0.266 * 2048**2, "linear ramp:  0.266·n²")
    for key, color, name, dy in (
        ("q=2.0", T["violet"], "gap²:  $n^{1.64}$", 0),
        ("q=1.5", T["orange"], "gap$^{1.5}$:  1.5·n", 12),
        ("q=2.0 tapered, n to 2048", T["aqua"], "gap², slowed ends:  1.25·n", -3),
    ):
        ax.plot(D[key]["n"], D[key]["p_half"], color=color, lw=2, zorder=3, **dots(T, color))
        end_label(ax, T, D[key]["n"][-1], D[key]["p_half"][-1], name, dy)
    ax.plot(n, n / 2, color=T["muted"], lw=1.4, zorder=2)
    end_label(ax, T, 2048, 1024, "exact schedule:  n/2", -10)
    ax.set_xlim(54, 2400)
    ax.set_ylim(20, 3e7)
    integer_ticks(ax.xaxis, [64, 128, 256, 512, 1024, 2048])
    integer_ticks(ax.yaxis, [100, 1000, 10000, 100000, 1000000])
    legend(
        ax,
        T,
        [
            Line2D(
                [], [], color=T["blue"], lw=2, label="linear ramp (LR-QAOA)", **dots(T, T["blue"])
            ),
            Line2D(
                [],
                [],
                color=T["violet"],
                lw=2,
                label="speed ∝ gap² (Roland–Cerf)",
                **dots(T, T["violet"]),
            ),
            Line2D(
                [], [], color=T["orange"], lw=2, label="speed ∝ gap^1.5", **dots(T, T["orange"])
            ),
            Line2D(
                [],
                [],
                color=T["aqua"],
                lw=2,
                label="speed ∝ gap², slowed at both ends",
                **dots(T, T["aqua"]),
            ),
            Line2D([], [], color=T["muted"], lw=1.4, label="closed-form exact schedule"),
        ],
        "upper left",
    )
    return fig


def q1_collapse(R):
    C = R["q1_tapered_curve"]
    fig, ax = new_axes()
    style(
        ax,
        T,
        "Success depends only on layers ÷ n",
        "speed ∝ gap², slowed at both ends; ring, Δγ = 0.6, Δβ = 0.3",
        "layers ÷ n",
        "P(x*)",
    )
    for color, n in zip(RAMP3, ("100", "512", "2048")):
        x = sorted(C[n], key=float)
        ax.plot(
            [float(v) for v in x],
            [C[n][v] for v in x],
            color=color,
            lw=2,
            zorder=3,
            **dots(T, color),
        )
    ax.plot([1.25, 1.25], [0, 0.505], color=T["ink2"], lw=0.9, zorder=1)
    ax.annotate(
        "P = 0.50 at 1.25·n",
        (1.25, 0.505),
        xytext=(1.55, 0.33),
        textcoords="data",
        color=T["ink2"],
        fontsize=9.3,
        va="center",
        ha="left",
    )
    end_label(ax, T, 5.0, C["2048"]["5.0"], "0.995 at 5·n", -10)
    ax.set_xlim(0.3, 5.2)
    ax.set_ylim(0, 1.03)
    ax.set_xticks([0.5, 1, 1.5, 2, 3, 5])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    legend(
        ax,
        T,
        [
            Line2D([], [], color=c, lw=2, label=f"n = {n}", **dots(T, c))
            for c, n in zip(RAMP3, (100, 512, 2048))
        ],
        "lower right",
    )
    return fig


def limit_single_flip(R):
    rows = [r for r in R["q6_threshold"] if r["graph"] != "torus 4x4"]
    regular = {"ring", "hypercube", "complete bipartite", "Petersen", "ramanujan"}
    fig, ax = new_axes(right=0.80)
    style(
        ax,
        T,
        "Past Δγ·h = π, more layers make things worse",
        "13 graphs, n = 8 to 16, Δβ = 0.3; h = largest local field in the best cut",
        "Δγ ÷ (π / h)",
        "P(x*) at 400 layers ÷ P(x*) at 50 layers",
    )
    ax.plot([0.55, 1.3], [1, 1], color=T["ink2"], lw=0.9, zorder=1)
    ax.plot([1, 1], [0, 4], color=T["ink2"], lw=0.9, zorder=1)
    for r in rows:
        color = T["blue"] if r["graph"] in regular else T["orange"]
        # at 1.5 both probabilities are ~0, the ratio is noise
        keep = [i for i, f in enumerate(r["factors"]) if f <= 1.25]
        ax.plot(
            [r["factors"][i] for i in keep],
            [r["ratio"][i] for i in keep],
            color=color,
            lw=1.5,
            alpha=0.8,
            zorder=3,
            **dots(T, color, 5.0),
        )
    ax.annotate(
        "more layers\nhelp",
        (1.3, 1.0),
        xytext=(8, 16),
        textcoords="offset points",
        color=T["ink2"],
        fontsize=9.3,
        va="center",
        annotation_clip=False,
    )
    ax.annotate(
        "more layers\nhurt",
        (1.3, 1.0),
        xytext=(8, -16),
        textcoords="offset points",
        color=T["ink2"],
        fontsize=9.3,
        va="center",
        annotation_clip=False,
    )
    ax.set_xlim(0.55, 1.3)
    ax.set_ylim(0, 4)
    ax.set_xticks([0.6, 0.8, 0.9, 1.0, 1.1, 1.25])
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    legend(
        ax,
        T,
        [
            Line2D(
                [],
                [],
                color=T["blue"],
                lw=1.5,
                label="unweighted regular (7)",
                **dots(T, T["blue"], 5.0),
            ),
            Line2D(
                [],
                [],
                color=T["orange"],
                lw=1.5,
                label="weighted or disordered (6)",
                **dots(T, T["orange"], 5.0),
            ),
        ],
        "upper left",
    )
    return fig


def limit_two_flip(R):
    rows = {(r["graph"], r["n"], r["seed"]): r for r in R["q6_threshold_deep"]}
    panels = (
        (("ramanujan", 14, 0), "random 3-regular, n = 14", 3),
        (("hypercube", 16, 0), "4-dimensional hypercube, n = 16", 4),
        (("complete bipartite", 12, 0), "complete bipartite, n = 12", 6),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.5), dpi=170)
    fig.patch.set_facecolor(T["surface"])
    fig.subplots_adjust(left=0.055, right=0.93, top=0.80, bottom=0.14, wspace=0.22)
    for ax, (key, name, d) in zip(axes, panels):
        r = rows[key]
        x = sorted(r["P"], key=float)
        style(
            ax,
            T,
            name,
            f"two adjacent flips wrap at {d / (2 * d - 2):.2f} of the single-flip value",
            "Δγ ÷ (π / h)",
            "P(x*)" if ax is axes[0] else "",
        )
        ax.axvspan(d / (2 * d - 2), 1.0, color=T["grid"], alpha=0.55, lw=0, zorder=0)
        ax.plot([d / (2 * d - 2)] * 2, [0, 1.03], color=T["ink2"], lw=0.9, zorder=1)
        for i, (color, depth) in enumerate(zip(RAMP3, (400, 1600, 6400))):
            y = [r["P"][f][i] for f in x]
            ax.plot([float(f) for f in x], y, color=color, lw=2, zorder=3, **dots(T, color))
            if ax is axes[-1]:
                end_label(ax, T, float(x[-1]), y[-1], f"{depth:,} layers")
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0, 1.03)
        ax.set_xticks([0.55, 0.7, 0.8, 0.9, 0.95])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    legend(
        axes[0],
        T,
        [
            Line2D([], [], color=c, lw=2, label=f"{p:,} layers", **dots(T, c))
            for c, p in zip(RAMP3, (400, 1600, 6400))
        ],
        "lower left",
    )
    return fig


def limit_collective(R):
    rows = {(r["db"], r["n"]): r for r in R["q6_complete_graph_stability"]}
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.5), dpi=170)
    fig.patch.set_facecolor(T["surface"])
    fig.subplots_adjust(left=0.055, right=0.93, top=0.80, bottom=0.14, wspace=0.22)
    for ax, db in zip(axes, (0.3, 0.6, 1.0)):
        f0 = rows[(db, 128)]
        style(
            ax,
            T,
            f"Δβ = {db}",
            f"formula: Δγ·n = {f0['predicted'] * 128:.1f}  (Δγ·Δβ·n = {f0['predicted'] * 128 * db:.1f})",
            "Δγ ÷ (collective formula)",
            "P(x*) at 400 layers" if ax is axes[0] else "",
        )
        ax.axvspan(1.1, 1.5, color=T["grid"], alpha=0.55, lw=0, zorder=0)
        ax.plot([1, 1], [0, 0.7], color=T["ink2"], lw=0.9, zorder=1)
        for color, n in zip(RAMP3, (128, 256, 512)):
            r = rows[(db, n)]
            ax.plot(r["factors"], r["P400"], color=color, lw=2, zorder=3, **dots(T, color))
        ax.set_xscale("log")
        ax.set_xlim(0.56, 2.15)
        ax.set_ylim(0, 0.7)
        ax.set_xticks([0.6, 0.8, 1.0, 1.25, 1.5, 2.0])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_yticks([0, 0.2, 0.4, 0.6])
    legend(
        axes[0],
        T,
        [
            Line2D([], [], color=c, lw=2, label=f"n = {n}", **dots(T, c))
            for c, n in zip(RAMP3, (128, 256, 512))
        ],
        "lower left",
    )
    return fig


def weighted_cliff(R):
    rows = R["q5_weighted_complete_beta_dependence"]
    S = R["q5_weighted_complete_cliffs"]
    dbs = (0.3, 0.6, 1.0)
    fig, ax = new_axes(right=0.74)
    style(
        ax,
        T,
        "On weighted complete graphs the limit moves with Δβ",
        "8 graphs, n = 14 and 16, positive weights; cliff = Δγ where P(x*) at 400 layers halves",
        "mixer step Δβ",
        "Δγ at the cliff",
    )
    med = lambda key, db: float(np.median([r[key] for r in rows if r["db"] == db]))
    ax.plot(dbs, [med("local", db) for db in dbs], color=T["orange"], lw=2, zorder=2)
    end_label(ax, T, 1.0, med("local", 1.0), "single-flip\nvalue π / h", 14)
    ax.plot(dbs, [med("collective", db) for db in dbs], color=T["aqua"], lw=2, zorder=2)
    end_label(ax, T, 1.0, med("collective", 1.0), "collective formula,\nmean weight", 3)
    for db in dbs:
        ax.plot(
            [db] * 8,
            [r["cliff"] for r in rows if r["db"] == db],
            ls="none",
            marker="o",
            ms=4.5,
            mfc=T["muted"],
            mec=T["surface"],
            mew=1.0,
            alpha=0.9,
            zorder=3,
        )
    ax.plot(
        dbs,
        [S[str(db)]["cliff"][1] for db in dbs],
        color=T["blue"],
        lw=2,
        zorder=4,
        **dots(T, T["blue"]),
    )
    end_label(ax, T, 1.0, S["1.0"]["cliff"][1], "measured\n(median)", -24)
    ax.set_xlim(0.22, 1.05)
    ax.set_ylim(0, 2.2)
    ax.set_xticks(dbs)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    legend(
        ax,
        T,
        [
            Line2D(
                [], [], color=T["blue"], lw=2, label="measured cliff, median", **dots(T, T["blue"])
            ),
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                ms=4.5,
                mfc=T["muted"],
                mec=T["surface"],
                label="each graph",
            ),
            Line2D([], [], color=T["orange"], lw=2, label="single-flip value"),
            Line2D([], [], color=T["aqua"], lw=2, label="collective formula"),
        ],
        "upper right",
    )
    return fig


def dehn_fit(R):
    rows = [r for r in R["q5_dehn_comparison"] if not r.get("printed")]
    N = [r["N"] for r in rows]
    fig, ax = new_axes(right=0.74)
    style(
        ax,
        T,
        "A published fit sits at 0.85 of the collective limit",
        "Dehn et al. 2026, complete graphs, uniform weights; limit uses mean weight 0.5",
        "qubits N",
        "Δγ",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(N, [r["threshold"] for r in rows], color=T["aqua"], lw=2, zorder=3)
    end_label(ax, T, N[-1], rows[-1]["threshold"], "collective limit\nat their Δβ", 6)
    ax.plot(N, [r["dg"] for r in rows], color=T["blue"], lw=2, zorder=3)
    end_label(ax, T, N[-1], rows[-1]["dg"], "published fit\n5.945·N^(−0.80)", -8)
    ax.axvspan(12, 28, color=T["grid"], alpha=0.55, lw=0, zorder=0)
    ax.annotate("range they fitted", (18.3, 0.19), color=T["ink2"], fontsize=9.3, ha="center")
    ax.set_xlim(11, 60)
    ax.set_ylim(0.16, 1.1)
    integer_ticks(ax.xaxis, [12, 16, 20, 28, 40, 56])
    ax.set_yticks([0.2, 0.3, 0.5, 1.0])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    legend(
        ax,
        T,
        [
            Line2D([], [], color=T["aqua"], lw=2, label="collective limit"),
            Line2D([], [], color=T["blue"], lw=2, label="published fit"),
        ],
        "upper right",
    )
    return fig


FIGURES_Q5 = dict(
    q1_depth=q1_depth,
    q1_collapse=q1_collapse,
    limit_single_flip=limit_single_flip,
    limit_two_flip=limit_two_flip,
    limit_collective=limit_collective,
    weighted_cliff=weighted_cliff,
    dehn_fit=dehn_fit,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["schedule_gap", "q5_q1"])
    a = ap.parse_args()
    plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]
    plt.rcParams["mathtext.default"] = "regular"
    if a.only in (None, "schedule_gap"):
        PANELS.mkdir(parents=True, exist_ok=True)
        N = collect()
        (PANELS / "numbers.json").write_text(json.dumps(N, indent=1))
        for p in render(N):
            print("wrote", p)
    if a.only in (None, "q5_q1"):
        Q51.mkdir(parents=True, exist_ok=True)
        R = json.loads(NUM.read_text())
        for name, draw in FIGURES_Q5.items():
            fig = draw(R)
            fig.savefig(Q51 / f"{name}.png", facecolor=T["surface"])
            plt.close(fig)
            print("wrote", Q51 / f"{name}.png")


if __name__ == "__main__":
    main()
