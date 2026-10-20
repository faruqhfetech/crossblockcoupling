"""Midterm figures: per-network rho by control, rho by training phase, dominance by depth.

  python -m analysis.figures --runs results/obs_mnist results/obs_cifar10 results/cnn_cifar10 --out results/midterm/figures
  add --compact for paper-sized figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import load as L
from . import local
from .stats import partial_rank_corr

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GROUPS = {"MLP · MNIST": "#2a78d6", "MLP · CIFAR-10": "#eb6834", "CNN · CIFAR-10": "#1baf7a", "CNN+skip · CIFAR-10": "#eda100"}
CONTROLS = {"none": "none", "position": "position", "overfit_gap": "+ overfit gap", "update_norm": "+ update norm",
            "residual": "+ residual", "persistence": "+ persistence"}
PHASES = ["early", "middle", "late"]

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})


def group(df):
    skip = df.get("skip", False).astype(str) == "True"
    mlp_group = np.where(df.dataset == "mnist", "MLP · MNIST", "MLP · CIFAR-10")
    cnn_group = np.where(skip, "CNN+skip · CIFAR-10", "CNN · CIFAR-10")
    return np.where(df.arch == "mlp", mlp_group, cnn_group)


def legend_handles(width):
    return [plt.Line2D([], [], color=color, lw=width, label=name) for name, color in GROUPS.items()]


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def network_meta(df):
    return df.groupby("network")[["arch", "dataset", "skip"]].first().reset_index()


def rho_by_control(df, out, compact=False):
    per = local.per_run(df, ["coupling"], local.CONTROLS)
    net = per.groupby(["network", "control"], as_index=False).rho.mean().merge(network_meta(df), on="network")
    net["group"] = group(net)

    fig, ax = plt.subplots(figsize=(6.5, 2.6) if compact else (9, 3.6))
    rng = np.random.default_rng(0)
    width = 0.18
    for i, control in enumerate(CONTROLS):
        for j, (name, color) in enumerate(GROUPS.items()):
            rhos = net[(net.control == control) & (net.group == name)].rho.dropna().to_numpy()
            x = i + (j - 1.5) * width
            jitter = rng.uniform(-width / 3, width / 3, len(rhos))
            ax.scatter(x + jitter, rhos, s=10, color=color, alpha=0.55, linewidths=0)
            ax.plot([x - width / 2.4, x + width / 2.4], [rhos.mean()] * 2, color=color, lw=2.2,
                    solid_capstyle="round")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(CONTROLS)), CONTROLS.values())
    ax.set_ylabel("within-network Spearman ρ\n(coupling vs next test-loss change)")
    if not compact:
        ax.set_title("Per-network ρ by control (dots = networks, bar = mean; ρ < 0: coupling precedes improvement)",
                     loc="left", fontsize=9.5)
    ax.legend(handles=legend_handles(2.2), frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, out, "rho_by_control")
    net.to_csv(out / "rho_by_control.csv", index=False)


def rho_by_phase(df, out, compact=False):
    rows = []
    for (network, _), g in df.groupby(["network", "split"]):
        g = g.sort_values("epoch")
        thirds = np.array_split(np.arange(len(g)), 3)
        for name, idx in zip(PHASES, thirds):
            part = g.iloc[idx]
            rho = partial_rank_corr(part.coupling, part.next_dloss, None, min_n=3)
            rows.append(dict(network=network, phase=name, rho=rho))
    net = pd.DataFrame(rows).groupby(["network", "phase"], as_index=False).rho.mean()
    net = net.merge(network_meta(df), on="network")
    net["group"] = group(net)

    fig, ax = plt.subplots(figsize=(3.3, 2.5) if compact else (5.2, 3.4))
    for j, (name, color) in enumerate(GROUPS.items()):
        means, lows, highs = [], [], []
        for phase in PHASES:
            rhos = net[(net.group == name) & (net.phase == phase)].rho.dropna()
            se = rhos.std(ddof=1) / np.sqrt(len(rhos))
            means.append(rhos.mean())
            lows.append(rhos.mean() - 1.96 * se)
            highs.append(rhos.mean() + 1.96 * se)
        x = np.arange(3) + (j - 1.5) * 0.06
        ax.vlines(x, lows, highs, color=color, lw=1.2, alpha=0.8)
        ax.plot(x, means, color=color, lw=2, marker="o", ms=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        if not compact:
            ax.annotate(name, (x[-1], means[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    if compact:
        ax.set_xticks(range(3), PHASES)
        ax.set_xlim(-0.3, 2.3)
    else:
        ax.set_xticks(range(3), ["early third", "middle third", "late third"])
        ax.set_xlim(-0.3, 2.9)
    ax.set_ylabel("within-network Spearman ρ")
    if compact:
        ax.legend(handles=legend_handles(2), frameon=False, fontsize=6, loc="lower left")
    else:
        ax.set_title("Per-network ρ by training phase (mean ± 95% CI over networks)", loc="left", fontsize=9.5)
    save(fig, out, "rho_by_phase")
    net.to_csv(out / "rho_by_phase.csv", index=False)


def abs_median(v):
    return v.abs().median()


def dominance_by_depth(df, out, compact=False):
    d = df.assign(group=group(df)).groupby(["group", "split", "depth"]).agg(
        a=("emp_delta_a", abs_median), b=("emp_delta_b", abs_median),
        balance=("n_params_a", "first")).reset_index()
    d["dominance"] = d.a / d.b

    panels = [("depth_fraction", "split at half the layers")]
    if not compact:
        panels.append(("param_count", "split at equal parameter count"))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3, 2.5) if compact else (9, 3.4), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (split, title) in zip(axes, panels):
        for name, color in GROUPS.items():
            series = d[(d.group == name) & (d.split == split)].sort_values("depth")
            ax.plot(series.depth, series.dominance, color=color, lw=2, marker="o", ms=6, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=name)
        ax.axhline(1, color=MUTED, lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_yticks([0.2, 0.3, 0.5, 1, 2], ["0.2", "0.3", "0.5", "1", "2"])
        ax.minorticks_off()
        ax.set_xticks(sorted(d.depth.unique()))
        ax.set_xlabel("depth (hidden layers / conv blocks)")
        if not compact:
            ax.set_title(title, loc="left", fontsize=9.5)

    if compact:
        axes[0].set_ylabel("median |Δ_a| / |Δ_b|  (log)")
        axes[0].legend(frameon=False, fontsize=6, loc="lower right")
    else:
        axes[0].set_ylabel("median |Δ_a| / |Δ_b|  (log)\n> 1: lower block dominates")
        fig.suptitle("Which block moves the upper Fisher trace more depends on the split", x=0.06, ha="left",
                     fontsize=10)
        axes[0].legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(1.05, -0.2))
    save(fig, out, "dominance_by_depth")
    d.to_csv(out / "dominance_by_depth.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compact", action="store_true", help="paper-sized figures, single-panel dominance")
    a = ap.parse_args()
    if a.compact:
        plt.rcParams.update({"font.size": 7.5})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df, _ = L.healthy(L.load(*a.runs))
    rho_by_control(df, out, a.compact)
    rho_by_phase(df, out, a.compact)
    dominance_by_depth(df, out, a.compact)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
