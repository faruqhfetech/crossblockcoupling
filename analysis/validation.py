"""Step-size linearity, actual vs control directions, horizons, probe reliability.

  python -m analysis.validation --input results/validation --out results/report_validation
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .stats import one_sample, partial_rank_corr

CK = ["run_id", "checkpoint_id"]
PROBE = ["probe_source", "probe_seed", "probe_size", "class_balanced"]
METRICS = ("signed_change", "absolute_change", "immediate_loss_change")


def near_zero_fit(alpha, change, radius=0.25):
    """Slope and R^2 of a line through the points with |alpha| <= radius."""
    near = np.abs(alpha) <= radius
    if near.sum() < 3 or len(np.unique(alpha[near])) < 2:
        return np.nan, np.nan
    x = alpha[near]
    y = change[near]
    slope, intercept = np.polyfit(x, y, 1)
    total_ss = ((y - y.mean()) ** 2).sum()
    if total_ss <= 0:
        return slope, np.nan
    resid_ss = ((y - slope * x - intercept) ** 2).sum()
    return slope, 1 - resid_ss / total_ss


def odd_symmetry_error(alpha, change):
    """Median of |f(a) + f(-a)| / (|f(a)| + |f(-a)|) over a > 0: 0 for an odd function."""
    by_alpha = dict(zip(np.round(alpha, 8), change))
    errors = []
    for a, value in by_alpha.items():
        if a > 0 and -a in by_alpha:
            mirror = by_alpha[-a]
            errors.append(abs(value + mirror) / max(abs(value) + abs(mirror), 1e-12))
    return np.median(errors) if errors else np.nan


def step_size(m):
    keys = CK + PROBE + ["direction_rep"]
    rows = []
    actual = m[m.direction == "actual"]
    for key_values, g in actual.groupby(keys):
        alpha = g.alpha.to_numpy(float)
        change = g.signed_change.to_numpy(float)
        slope, r2 = near_zero_fit(alpha, change)
        rows.append(dict(zip(keys, key_values), slope=slope, r2=r2,
                         odd_symmetry_error=odd_symmetry_error(alpha, change)))
    return pd.DataFrame(rows)


def directions(m):
    at_one = m[np.isclose(m.alpha, 1)]
    means = at_one.groupby(CK + PROBE + ["direction"], as_index=False)[list(METRICS)].mean()
    actual = means[means.direction == "actual"].drop(columns="direction")
    controls = means[means.direction != "actual"]
    pairs = controls.merge(actual, on=CK + PROBE, suffixes=("_control", "_actual"))
    rows = []
    for direction, g in pairs.groupby("direction"):
        for metric in METRICS:
            difference = g[f"{metric}_actual"] - g[f"{metric}_control"]
            rows.append(dict(direction=direction, metric=metric, **one_sample(difference)))
    return pd.DataFrame(rows)


def horizons(m, h):
    primary_rows = m[(m.direction == "actual") & np.isclose(m.alpha, 1)]
    primary = primary_rows.groupby(CK, as_index=False).agg(
        signed=("signed_change", "mean"), absolute=("absolute_change", "mean"))
    hh = h.merge(primary, on=CK, how="left")
    hh["horizon"] = hh.horizon_steps.astype(int)
    rows = []
    for (horizon, arch), g in hh.groupby(["horizon", "arch"]):
        for predictor in ("signed", "absolute"):
            rhos = []
            for _, run in g.groupby("run_id"):
                rhos.append(partial_rank_corr(run[predictor], run.test_loss_change))
            rows.append(dict(horizon_steps=horizon, arch=arch, predictor=predictor, **one_sample(rhos)))
    return pd.DataFrame(rows)


def reliability(m):
    at_one = m[(m.direction == "actual") & np.isclose(m.alpha, 1)]
    rows = []
    for factor in PROBE:
        rest = [c for c in PROBE if c != factor]
        wide = at_one.pivot_table(index=CK + rest, columns=factor, values="signed_change", aggfunc="mean")
        wide.columns = [str(c) for c in wide.columns]  # bool/int labels would be read as masks
        levels = list(wide.columns)
        for i, level_a in enumerate(levels):
            for level_b in levels[i + 1:]:
                n_pairs = int(wide[[level_a, level_b]].dropna().shape[0])
                rows.append(dict(factor=factor, level_a=level_a, level_b=level_b,
                                 rho=partial_rank_corr(wide[level_a], wide[level_b]), n=n_pairs))
    rel = pd.DataFrame(rows)

    same_probe = CK + ["probe_source", "probe_seed", "class_balanced"]
    largest = at_one[at_one.probe_size == at_one.probe_size.max()][same_probe + ["signed_change"]]
    conv = at_one.merge(largest, on=same_probe, suffixes=("", "_ref"))
    error = (conv.signed_change - conv.signed_change_ref).abs()
    conv["rel_err"] = error / conv.signed_change_ref.abs().clip(lower=1e-12)
    return rel, conv.groupby("probe_size", as_index=False).rel_err.median()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = Path(a.input)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(src / "validation_measurements.csv")
    h = pd.read_csv(src / "validation_horizons.csv")

    lines = []
    zero_change = m[np.isclose(m.alpha, 0)].signed_change.abs().max()
    n_actual = int(((m.direction == "actual") & np.isclose(m.alpha, 1)).sum())
    lines += [f"A. alpha=0 max |change| = {zero_change:.2e}; actual alpha=1 rows = {n_actual}", ""]

    fits = step_size(m)
    fits.to_csv(out / "step_size_fits.csv", index=False)
    lines += [f"B. step-size: median near-zero R^2 {fits.r2.median():.4f}, "
              f"median odd-symmetry error {fits.odd_symmetry_error.median():.4f}", ""]

    dirs = directions(m)
    dirs.to_csv(out / "directions.csv", index=False)
    lines += ["C. ACTUAL minus CONTROL at alpha=1", dirs.round(5).to_string(index=False), ""]

    hz = horizons(m, h)
    hz.to_csv(out / "horizons.csv", index=False)
    lines += ["D. HORIZONS: within-run rho with test-loss change", hz.round(4).to_string(index=False), ""]

    rel, conv = reliability(m)
    rel.to_csv(out / "probe_reliability.csv", index=False)
    conv.to_csv(out / "probe_size_convergence.csv", index=False)
    lines += ["E. PROBE RELIABILITY (median pairwise rho)",
              rel.groupby("factor").rho.median().round(4).to_string(),
              "relative error vs largest probe, by size", conv.round(4).to_string(index=False)]
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
