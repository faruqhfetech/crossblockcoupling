"""Final accuracy, seed-pair concordance and paired condition tests (no test-set selection)."""
from itertools import combinations

import pandas as pd
from scipy import stats

from .stats import paired


def endpoints(df) -> pd.DataFrame:
    """One row per network, from its first split's series."""
    first_split = df.groupby("network").split.transform("min")
    rows = []
    for net, g in df[df.split == first_split].groupby("network"):
        g = g.sort_values("epoch")
        last = g.iloc[-1]
        row = dict(network=net, dataset=g.dataset.iloc[0], arch=g.arch.iloc[0],
                   condition=g.condition.iloc[0], depth=g.depth.iloc[0], seed=g.seed.iloc[0],
                   final_test_acc=last.test_acc, final_test_loss=last.test_loss,
                   mean_coupling=g.coupling.mean())
        if "val_loss" in g and g.val_loss.notna().any():
            best_val = g.val_loss.to_numpy().argmin()
            row["selected_test_acc"] = g.test_acc.iloc[best_val]
        rows.append(row)
    return pd.DataFrame(rows)


def concordance(ep: pd.DataFrame, x="mean_coupling", y="final_test_acc"):
    """Fixed condition and depth: does the higher-x seed also have higher y?
    Pairs within a cell share depth and condition, so they are not independent;
    the binomial p is descriptive."""
    rows = []
    for (dataset, condition), g in ep.groupby(["dataset", "condition"]):
        concordant = 0
        discordant = 0
        for _, cell in g.groupby("depth"):
            for (_, a), (_, b) in combinations(cell.iterrows(), 2):
                if a[x] == b[x] or a[y] == b[y]:
                    continue
                same_order = (a[x] > b[x]) == (a[y] > b[y])
                if same_order:
                    concordant += 1
                else:
                    discordant += 1
        pairs = concordant + discordant
        if pairs >= 6:
            p_binom = stats.binomtest(concordant, pairs, 0.5).pvalue
            rows.append(dict(dataset=dataset, condition=condition, pairs=pairs, frac=concordant / pairs,
                             p_binom=p_binom))
    return pd.DataFrame(rows)


def paired_conditions(ep: pd.DataFrame, treatment: str, baselines, outcome="final_test_acc",
                      pair_on=("dataset", "depth", "seed")):
    """Treatment minus each baseline condition, paired on `pair_on`."""
    wide = ep.pivot_table(index=list(pair_on), columns="condition", values=outcome, aggfunc="first")
    rows = []
    for baseline in baselines:
        if treatment not in wide or baseline not in wide:
            continue
        both = wide[[treatment, baseline]].dropna()
        rows.append(dict(treatment=treatment, baseline=baseline, outcome=outcome,
                         **paired(both[treatment], both[baseline])))
    return pd.DataFrame(rows)
