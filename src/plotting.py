"""Figures for the report (purple visual identity)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .experiments import summarise  # noqa: E402

COLORS = {"GA": "#5B2A86", "PSO": "#C2378F", "ACO": "#E09F1F", "HGA": "#16A3A0",
          "DP": "#1F1530", "Greedy": "#8A8796"}
META = ["GA", "PSO", "ACO", "HGA"]
SIZE_ORDER = ["small", "medium", "large"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": "#B9A9CF", "axes.labelcolor": "#2D1B45",
    "axes.titleweight": "bold", "axes.titlecolor": "#2D1B45",
    "axes.grid": True, "grid.color": "#ECE6F4", "grid.linewidth": 0.8,
    "xtick.color": "#4A3A63", "ytick.color": "#4A3A63",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 200,
})


def _padded(histories: list[list[int]]) -> np.ndarray:
    """Pad histories (early stopping) with their last value -> equal length."""
    L = max(len(h) for h in histories)
    return np.array([h + [h[-1]] * (L - len(h)) for h in histories], dtype=float)


def plot_convergence(report: dict, fig_dir: Path) -> None:
    sizes = [s for s in SIZE_ORDER if s in report["instances"]]
    fig, axes = plt.subplots(len(sizes), 1, figsize=(8.5, 3.1 * len(sizes)))
    axes = np.atleast_1d(axes)
    for ax, size in zip(axes, sizes):
        entry = report["instances"][size]
        opt = entry["optimum"]
        for alg in META:
            H = _padded([r["history"] for r in entry["algorithms"][alg]["runs"]])
            m, s = H.mean(0), H.std(0)
            x = np.arange(H.shape[1])
            ax.plot(x, m, color=COLORS[alg], lw=2, label=alg)
            ax.fill_between(x, m - s, m + s, color=COLORS[alg], alpha=0.15, lw=0)
        ax.axhline(opt, color=COLORS["DP"], ls="--", lw=1.2, label="DP optimum")
        g = entry["algorithms"]["Greedy"]["runs"][0]["best_value"]
        ax.axhline(g, color=COLORS["Greedy"], ls=":", lw=1.4, label="Greedy")
        lo = min(np.min(_padded([r["history"] for r in entry["algorithms"][a]["runs"]])[:, 0])
                 for a in META)
        ax.set_ylim(lo - 0.02 * opt, opt + 0.01 * opt)
        ax.set_title(f"{size.capitalize()} (n={entry['n']})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best fitness (total return)")
    axes[0].legend(loc="lower right", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(fig_dir / "convergence.png")
    plt.close(fig)

    # Zoomed version: gap-to-optimum on log scale (shows late convergence)
    fig, axes = plt.subplots(len(sizes), 1, figsize=(8.5, 2.8 * len(sizes)))
    axes = np.atleast_1d(axes)
    for ax, size in zip(axes, sizes):
        entry = report["instances"][size]
        opt = entry["optimum"]
        for alg in META:
            H = _padded([r["history"] for r in entry["algorithms"][alg]["runs"]])
            gap = 100 * (opt - H.mean(0)) / opt
            ax.plot(np.maximum(gap, 1e-3), color=COLORS[alg], lw=2, label=alg)
        ax.set_yscale("log")
        ax.set_title(f"Gap to optimum - {size} (n={entry['n']})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean gap (%)  [log]")
    axes[0].legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(fig_dir / "convergence_gap.png")
    plt.close(fig)


def plot_boxplots(report: dict, fig_dir: Path) -> None:
    sizes = [s for s in SIZE_ORDER if s in report["instances"]]
    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 3.6))
    axes = np.atleast_1d(axes)
    for ax, size in zip(axes, sizes):
        entry = report["instances"][size]
        opt = entry["optimum"]
        data = [[100 * (opt - r["best_value"]) / opt for r in entry["algorithms"][a]["runs"]]
                for a in META]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5)
        ax.set_xticks(range(1, len(META) + 1), META)
        for patch, alg in zip(bp["boxes"], META):
            patch.set_facecolor(COLORS[alg]); patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color("white"); med.set_linewidth(2)
        ax.set_title(f"Stability - {size} (n={entry['n']})")
        ax.set_ylabel("Gap to optimum (%)")
    fig.tight_layout()
    fig.savefig(fig_dir / "stability_boxplot.png")
    plt.close(fig)


def _grouped_bars(rows, metric, ylabel, title, fname, fig_dir, algs, log=False):
    sizes = [s for s in SIZE_ORDER if any(r["size"] == s for r in rows)]
    x = np.arange(len(sizes))
    width = 0.8 / len(algs)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for k, alg in enumerate(algs):
        vals = [next(r[metric] for r in rows if r["size"] == s and r["algorithm"] == alg)
                for s in sizes]
        ax.bar(x + (k - (len(algs) - 1) / 2) * width, vals, width,
               color=COLORS[alg], label=alg)
    ax.set_xticks(x, [s.capitalize() for s in sizes])
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=len(algs), fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / fname)
    plt.close(fig)


def plot_bars(report: dict, fig_dir: Path) -> None:
    rows = summarise(report)
    all_algs = ["DP", "Greedy", *META]
    _grouped_bars(rows, "mean_time_s", "Mean run time (s) [log]",
                  "Execution time", "time.png", fig_dir, all_algs, log=True)
    _grouped_bars(rows, "mean_gap_pct", "Mean gap to optimum (%)",
                  "Solution quality (lower is better)", "gap.png", fig_dir,
                  ["Greedy", *META])
    _grouped_bars(rows, "memory_mb", "Peak memory (MB) [log]",
                  "Peak memory usage", "memory.png", fig_dir, all_algs, log=True)


def plot_sensitivity(sens: dict, fig_dir: Path) -> None:
    algs = list(sens["results"])
    fig, axes = plt.subplots(1, len(algs), figsize=(4.6 * len(algs), 3.4))
    for ax, alg in zip(axes, algs):
        d = sens["results"][alg]
        xs = [f"{p['value']:.3g}" for p in d["points"]]
        ys = [p["gap_pct"] for p in d["points"]]
        ax.bar(xs, ys, color=COLORS[alg], alpha=0.85)
        ax.set_title(f"{alg}: {d['param']}")
        ax.set_xlabel(d["param"])
        ax.set_ylabel("Mean gap (%)")
    fig.tight_layout()
    fig.savefig(fig_dir / "sensitivity.png")
    plt.close(fig)


def plot_dp_scaling(scal: dict, fig_dir: Path) -> None:
    pts = scal["points"]
    C = [p["capacity"] for p in pts]
    fig, ax1 = plt.subplots(figsize=(7.5, 3.6))
    ax1.plot(C, [p["memory_mb"] for p in pts], "o-", color=COLORS["GA"], lw=2, label="Peak memory (MB)")
    ax1.set_xlabel("Budget C (finer cost units)")
    ax1.set_ylabel("Peak memory (MB)", color=COLORS["GA"])
    ax2 = ax1.twinx()
    ax2.plot(C, [p["time"] for p in pts], "s--", color=COLORS["ACO"], lw=2, label="Time (s)")
    ax2.set_ylabel("Time (s)", color=COLORS["ACO"])
    ax2.grid(False)
    ax1.set_title(f"DP scaling with budget size (n={scal['n']})")
    fig.tight_layout()
    fig.savefig(fig_dir / "dp_scaling.png")
    plt.close(fig)


def make_all(report: dict, sens: dict | None, fig_dir: Path,
             scal: dict | None = None) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_convergence(report, fig_dir)
    plot_boxplots(report, fig_dir)
    plot_bars(report, fig_dir)
    if sens:
        plot_sensitivity(sens, fig_dir)
    if scal:
        plot_dp_scaling(scal, fig_dir)
