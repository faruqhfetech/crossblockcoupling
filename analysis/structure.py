"""Dominance, residuals, F_aa/F_bb, spectrum, emp vs model Fisher, cross-talk, rank dynamics,
moderators, run-level aggregations and robustness.

  python -m analysis.structure --runs results/RUNS --out results/report_structure
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import load as L
from . import local
from .stats import bh_fdr, one_sample, partial_rank_corr

CELL = ["dataset", "split", "condition", "depth"]
POSITION = {"position": ["progress"]}


def abs_median(v):
    return v.abs().median()


def dominance(d):
    g = d.groupby(CELL).agg(abs_a=("emp_delta_a", abs_median), abs_b=("emp_delta_b", abs_median),
                            n_params_a=("n_params_a", "first"), n_params_b=("n_params_b", "first"),
                            split_cut=("split_cut", "first")).reset_index()
    g["dominance"] = g.abs_a / g.abs_b.replace(0, np.nan)
    g["param_balance_a_over_b"] = g.n_params_a / g.n_params_b
    return g


def residual(d):
    medians = d.groupby(CELL).resid_frac.median().reset_index()
    trend_rows = []
    for (dataset, split, condition), s in medians.groupby(["dataset", "split", "condition"]):
        if s.depth.nunique() > 2:
            rho = stats.spearmanr(s.depth, s.resid_frac)[0]
        else:
            rho = np.nan
        trend_rows.append(dict(dataset=dataset, split=split, condition=condition, rho_depth=rho))
    return medians, pd.DataFrame(trend_rows)


def emp_vs_model(d):
    pairs = [("emp_trace_b", "model_trace_b"), ("emp_delta_a", "model_delta_a"), ("emp_delta_b", "model_delta_b")]
    rows = []
    for (dataset, condition), g in d.groupby(["dataset", "condition"]):
        row = dict(dataset=dataset, condition=condition, n=len(g))
        for emp, model in pairs:
            if model in g and g[[emp, model]].dropna().shape[0] >= 10:
                quantity = emp[4:]
                row[f"rho_{quantity}"] = stats.spearmanr(g[emp], g[model], nan_policy="omit")[0]
        rows.append(row)
    return pd.DataFrame(rows)


def epoch_trend(g):
    if g.crosstalk.notna().sum() < 4:
        return np.nan
    return stats.spearmanr(g.epoch, g.crosstalk)[0]


def crosstalk(d):
    fab = np.sqrt(d.emp_Fab_frob_sq.clip(lower=0))
    scale = np.sqrt((d.emp_trace_a * d.emp_trace_b).clip(lower=0)).replace(0, np.nan)
    d = d.assign(crosstalk=fab / scale)
    level = d.groupby(CELL).crosstalk.median().reset_index()
    groups = d.groupby(["network", "split", "dataset", "condition"])
    trend = groups[["epoch", "crosstalk"]].apply(epoch_trend).rename("rho_epoch").reset_index()
    return level, trend, d


def rank_shape(rho_first_half, rho_second_half):
    if rho_first_half > 0 > rho_second_half:
        return "hump"
    if rho_first_half < 0 and rho_second_half < 0:
        return "collapse"
    if rho_first_half > 0 and rho_second_half > 0:
        return "growth"
    return "other"


def rank_dynamics(d):
    rows = []
    for (net, split), g in d.groupby(["network", "split"]):
        g = g.sort_values("epoch")
        if len(g) < 6:
            continue
        eff_rank = g.emp_effective_rank_b.to_numpy()
        half = len(g) // 2
        rho_first = stats.spearmanr(np.arange(half), eff_rank[:half])[0]
        rho_second = stats.spearmanr(np.arange(len(g) - half), eff_rank[half:])[0]
        rows.append(dict(network=net, split=split, dataset=g.dataset.iloc[0], condition=g.condition.iloc[0],
                         shape=rank_shape(rho_first, rho_second),
                         rho_overfit_gap=stats.spearmanr(g.overfit_gap, eff_rank)[0]))
    return pd.DataFrame(rows)


def growth(d):
    rows = []
    for (net, split), g in d.groupby(["network", "split"]):
        g = g.sort_values("epoch")
        rows.append(dict(network=net, split=split, dataset=g.dataset.iloc[0], condition=g.condition.iloc[0],
                         coupling_grew=g.coupling.iloc[-1] > g.coupling.iloc[0],
                         crosstalk_grew=g.crosstalk.iloc[-1] > g.crosstalk.iloc[0]))
    per_run = pd.DataFrame(rows)
    return per_run.groupby(["dataset", "condition"])[["coupling_grew", "crosstalk_grew"]].mean().reset_index()


MODERATORS = {"effective_rank": "emp_effective_rank_b", "residual_frac": "resid_frac", "position": "progress",
              "overfit_gap": "overfit_gap", "crosstalk": "crosstalk"}


def moderator_table(d):
    """rho(coupling, next_dloss) below vs above each run's median of the moderator."""
    rows = []
    for name, column in MODERATORS.items():
        rhos_low, rhos_high = [], []
        for _, g in d.groupby(["network", "split"]):
            g = g.dropna(subset=["coupling", "next_dloss", column])
            if len(g) < 6:
                continue
            high = g[column] > g[column].median()
            low = ~high
            if low.sum() >= 3:
                rhos_low.append(stats.spearmanr(g.coupling[low], g.next_dloss[low])[0])
            if high.sum() >= 3:
                rhos_high.append(stats.spearmanr(g.coupling[high], g.next_dloss[high])[0])
        rhos_low = [v for v in rhos_low if np.isfinite(v)]
        rhos_high = [v for v in rhos_high if np.isfinite(v)]
        if len(rhos_low) >= 10 and len(rhos_high) >= 10:
            rows.append(dict(moderator=name, rho_low=np.mean(rhos_low), rho_high=np.mean(rhos_high),
                             p=stats.ttest_ind(rhos_low, rhos_high).pvalue))
    table = pd.DataFrame(rows)
    if len(table):
        table["p_fdr"] = bh_fdr(table.p.to_numpy())
    return table


def moderators(d):
    screen = moderator_table(d[d.dataset == "mnist"])
    holdout = moderator_table(d[d.dataset == "cifar10"])
    if not len(screen) or not len(holdout):
        return pd.DataFrame()
    m = screen.merge(holdout, on="moderator", suffixes=("_mnist", "_cifar10"))
    direction_mnist = np.sign(m.rho_low_mnist - m.rho_high_mnist)
    direction_cifar = np.sign(m.rho_low_cifar10 - m.rho_high_cifar10)
    m["same_direction"] = direction_mnist == direction_cifar
    m["survives"] = m.same_direction & (m.p_fdr_mnist < 0.05) & (m.p_fdr_cifar10 < 0.05)
    return m


def partial_p(rho, n):
    """Two-sided p for a partial correlation with one control (n - 3 degrees of freedom)."""
    if not np.isfinite(rho):
        return np.nan
    t = rho * np.sqrt((n - 3) / max(1 - rho ** 2, 1e-12))
    return 2 * stats.t.sf(abs(t), n - 3)


def aggregations(d):
    first_split = d.groupby("network").split.transform("min")
    run_rows = []
    for net, g in d[d.split == first_split].groupby("network"):
        g = g.sort_values("epoch")
        coupling = g.coupling.to_numpy()
        k = max(2, len(g) // 3)
        run_rows.append(dict(network=net, dataset=g.dataset.iloc[0], condition=g.condition.iloc[0],
                             depth=g.depth.iloc[0], final_acc=g.test_acc.iloc[-1], mean=coupling.mean(),
                             early=coupling[:k].mean(), late=coupling[-k:].mean(), std=coupling.std(),
                             max=coupling.max(), trend=stats.spearmanr(np.arange(len(coupling)), coupling)[0]))
    runs = pd.DataFrame(run_rows)

    rows = []
    for (dataset, condition), g in runs.groupby(["dataset", "condition"]):
        if len(g) < 10:
            continue
        for aggregation in ("mean", "early", "late", "std", "max", "trend"):
            rho = partial_rank_corr(g[aggregation], g.final_acc, g[["depth"]].to_numpy())
            n = g[[aggregation, "final_acc"]].dropna().shape[0]
            rows.append(dict(dataset=dataset, condition=condition, aggregation=aggregation, n=n,
                             rho_partial_depth=rho, p=partial_p(rho, n)))
    table = pd.DataFrame(rows)
    if len(table):
        table["p_fdr"] = bh_fdr(table.p.fillna(1).to_numpy())
    return table


def position_rhos(df):
    return local.by_network(local.per_run(df, ["coupling"], POSITION)).rho


def robustness(df_all, clean):
    per_net = local.by_network(local.per_run(clean, ["coupling"], POSITION))
    loco = [dict(left_out="none", **one_sample(per_net.rho))]
    for condition in sorted(per_net.condition.unique()):
        loco.append(dict(left_out=condition, **one_sample(per_net[per_net.condition != condition].rho)))

    depths = clean.groupby("network")[["dataset", "condition", "depth"]].first().reset_index()
    with_depth = per_net.merge(depths[["network", "depth"]], on="network")
    cell_means = with_depth.groupby(["dataset", "condition", "depth"]).rho.mean()

    health = L.health(df_all)
    trained = health.network[~health.training_failed]
    variants = {"unfiltered": df_all, "chance_only": df_all[df_all.network.isin(trained)], "full_filter": clean}
    filter_rows = [dict(variant=name, **one_sample(position_rhos(v))) for name, v in variants.items()]
    return (pd.DataFrame(loco), pd.DataFrame([dict(unit="condition_cell", **one_sample(cell_means))]),
            pd.DataFrame(filter_rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = L.load(*a.runs)
    clean, _ = L.healthy(df)
    if clean.empty:
        raise SystemExit("no healthy networks (all at chance or unstable)")
    lines = []

    def emit(name, table, title):
        table.to_csv(out / f"{name}.csv", index=False)
        text = table.round(4).to_string(index=False) if len(table) else "(none)"
        lines.extend([title, text, ""])

    emit("dominance", dominance(clean), "DOMINANCE median |delta_a|/|delta_b|")
    residuals, residual_trend = residual(clean)
    emit("residual", residuals, "ADDITIVITY RESIDUAL FRACTION (median)")
    emit("residual_depth_trend", residual_trend, "RESIDUAL vs DEPTH")

    trace_ratio = clean.assign(r=clean.emp_trace_a / clean.emp_trace_b)
    faa_fbb = trace_ratio.groupby(CELL).r.median().rename("faa_over_fbb").reset_index()
    emit("faa_fbb", faa_fbb, "tr F_aa / tr F_bb (median)")

    spectrum = clean.groupby(["dataset", "condition"]).agg(
        top_eig=("emp_top_eigenvalue_b", "mean"), eff_rank=("emp_effective_rank_b", "mean")).reset_index()
    emit("spectrum", spectrum, "F_bb SPECTRUM")
    emit("emp_vs_model", emp_vs_model(clean), "EMPIRICAL vs MODEL-LABEL FISHER (Spearman)")

    crosstalk_level, crosstalk_trend, clean = crosstalk(clean)
    emit("crosstalk", crosstalk_level, "RELATIVE CROSS-TALK ||F_ab|| / sqrt(tr F_aa tr F_bb) (median)")
    trend_medians = crosstalk_trend.groupby(["dataset", "condition"]).rho_epoch.median().reset_index()
    emit("crosstalk_trend", trend_medians, "CROSS-TALK TREND (median within-run rho with epoch)")

    shapes = rank_dynamics(clean)
    if len(shapes):
        shapes = shapes.groupby(["dataset", "shape"]).size().unstack(fill_value=0).reset_index()
    emit("rank_dynamics", shapes, "EFFECTIVE-RANK SHAPE COUNTS")
    emit("growth", growth(clean), "FIRST vs LAST CHECKPOINT (fraction of runs that grew)")
    emit("moderators", moderators(clean), "MODERATORS (MNIST screen, CIFAR-10 holdout, BH-FDR)")
    emit("aggregations", aggregations(clean), "RUN-LEVEL AGGREGATIONS vs FINAL ACCURACY (depth-partial)")

    loco, cells, filters = robustness(df, clean)
    emit("leave_one_condition_out", loco, "LEAVE-ONE-CONDITION-OUT (position-controlled coupling rho)")
    emit("condition_cells", cells, "CONDITION-CELL INFERENCE")
    emit("filter_sensitivity", filters, "FILTER SENSITIVITY")
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
