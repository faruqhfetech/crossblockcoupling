"""Training profile of the stage 6 ViTs, broken down by training phase.

1. Per seed: when test loss bottoms out, and final loss and accuracy.
2. By training window: test loss, coupling, upper Fisher trace, which block changes the upper Fisher more,
   and the raw within-run correlation of coupling with the next test-loss change.
3. Before vs after each seed's best test loss: position-controlled correlation on each part.
4. Control ladder on the whole run: position; position + overfit gap; position + change in the overfit gap.
5. Position-controlled correlation within each training window.

Correlations are computed per (seed, block split) and averaged over the two splits, so the unit is the network.
Negative rho = higher coupling comes before more improvement.

  python -m analysis.vit_profile                # reads results/vit_cifar100, writes results/final/stage6/vit_profile.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .stats import one_sample, partial_rank_corr

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "vit_cifar100"
OUT = ROOT / "results" / "final" / "stage6" / "vit_profile.txt"
WINDOWS = [(1, 10), (11, 30), (31, 60), (61, 100)]
LADDER = {
    "position": ["epoch"],
    "position + overfit gap": ["epoch", "overfit_gap"],
    "position + change in overfit gap": ["epoch", "gap_change"],
    "position + gap + change in gap": ["epoch", "overfit_gap", "gap_change"],
}


def load():
    frames = [pd.read_csv(path) for path in sorted(RUNS.glob("*.csv"))]
    df = pd.concat(frames, ignore_index=True).sort_values(["seed", "split", "epoch"]).reset_index(drop=True)
    df["coupling"] = df.emp_delta_a.abs()
    by_series = df.groupby(["seed", "split"])
    df["next_dloss"] = by_series.test_loss.shift(-1) - df.test_loss
    df["overfit_gap"] = df.test_loss - df.train_loss_running
    df["gap_change"] = df.overfit_gap - by_series.overfit_gap.shift(1)
    best_epoch = df.loc[df.groupby("seed").test_loss.idxmin(), ["seed", "epoch"]].set_index("seed").epoch
    df["best_epoch"] = df.seed.map(best_epoch)
    return df


def network_rhos(df, controls):
    """One rho per seed: partial rho within each block split, averaged over splits."""
    rhos = []
    for seed, g in df.groupby("seed"):
        per_split = [partial_rank_corr(s.coupling, s.next_dloss, s[controls].to_numpy())
                     for _, s in g.groupby("split")]
        rhos.append(np.nanmean(per_split))
    return np.array(rhos)


def describe(rhos):
    summary = one_sample(rhos)
    positive = int(np.sum(rhos > 0))
    return dict(mean=round(summary["mean"], 3), positive=f"{positive}/{summary['n']}",
                p_wilcoxon=round(summary["p_wilcoxon"], 4))


def per_seed(df):
    one_split = df[df.split == "depth_fraction"]
    rows = []
    for seed, g in one_split.groupby("seed"):
        best = g.loc[g.test_loss.idxmin()]
        rows.append(dict(seed=seed, best_test_loss_epoch=int(best.epoch), best_test_loss=round(best.test_loss, 3),
                         final_test_loss=round(g.test_loss.iloc[-1], 3), best_acc=round(g.test_acc.max(), 3),
                         final_acc=round(g.test_acc.iloc[-1], 3)))
    return pd.DataFrame(rows)


def windows_profile(df):
    one_split = df[df.split == "depth_fraction"]
    rows = []
    for first, last in WINDOWS:
        part = one_split[one_split.epoch.between(first, last)]
        raw = [stats.spearmanr(g.coupling, g.next_dloss, nan_policy="omit")[0] for _, g in part.groupby("seed")]
        dominance = (part.emp_delta_a.abs() / part.emp_delta_b.abs()).median()
        rows.append(dict(epochs=f"{first}-{last}", mean_test_loss=round(part.test_loss.mean(), 3),
                         mean_coupling=round(part.coupling.mean(), 2),
                         share_delta_a_negative=round((part.emp_delta_a < 0).mean(), 2),
                         mean_upper_trace=round(part.emp_trace_b.mean(), 1),
                         median_lower_over_upper=round(dominance, 2),
                         share_test_loss_falls=round((part.next_dloss < 0).mean(), 2),
                         raw_rho_mean=round(np.nanmean(raw), 3),
                         raw_rho_positive=f"{int(np.sum(np.array(raw) > 0))}/{len(raw)}"))
    return pd.DataFrame(rows)


def before_after_best(df):
    rows = []
    parts = {"up to best test loss": df.epoch <= df.best_epoch, "after best test loss": df.epoch > df.best_epoch}
    for label, mask in parts.items():
        part = df[mask]
        rows.append(dict(part=label, epochs_per_run=round(part.groupby("seed").epoch.nunique().mean(), 1),
                         **describe(network_rhos(part, ["epoch"]))))
    return pd.DataFrame(rows)


def control_ladder(df):
    rows = []
    for label, controls in LADDER.items():
        rows.append(dict(controls=label, **describe(network_rhos(df, controls))))
    return pd.DataFrame(rows)


def windows_controlled(df):
    rows = []
    for first, last in WINDOWS:
        part = df[df.epoch.between(first, last)]
        for label in ("position", "position + change in overfit gap"):
            rows.append(dict(epochs=f"{first}-{last}", controls=label, **describe(network_rhos(part, LADDER[label]))))
    return pd.DataFrame(rows)


def main():
    df = load()
    sections = [
        ("1. PER SEED (depth_fraction split)", per_seed(df)),
        ("2. BY TRAINING WINDOW (depth_fraction split; raw within-run Spearman, coupling vs next test-loss change)",
         windows_profile(df)),
        ("3. BEFORE vs AFTER EACH SEED'S BEST TEST LOSS (position-controlled, both splits averaged)",
         before_after_best(df)),
        ("4. CONTROL LADDER, WHOLE RUN (both splits averaged)", control_ladder(df)),
        ("5. CONTROLLED, WITHIN EACH TRAINING WINDOW (both splits averaged)", windows_controlled(df)),
    ]
    text = "\n\n".join(f"{title}\n{table.to_string(index=False)}" for title, table in sections)
    text += "\n\nWith 5 networks the smallest possible two-sided Wilcoxon p is 0.0625."
    OUT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
