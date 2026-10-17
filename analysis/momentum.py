"""Finite-history momentum and gating. Improvement = L_t - L_t+1 (positive = coupling precedes learning).

  python -m analysis.momentum --runs results/RUNS --out results/report_momentum
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from . import load as L
from . import local
from .stats import one_sample, paired


def label(df):
    """'sgd' for K = 0, otherwise K<k>_b<beta>_<gate>."""
    K = df.history_K.astype(int)
    history_label = "K" + K.astype(str) + "_b" + df.history_beta.astype(str) + "_" + df.gate.astype(str)
    return np.where(K == 0, "sgd", history_label)


def clustered(formula, d):
    """OLS with standard errors clustered by network."""
    used_columns = [c for c in d.columns if c in formula]
    z = d.dropna(subset=used_columns)
    clusters = pd.factorize(z.network)[0]
    fit = smf.ols(formula, data=z).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    table = pd.DataFrame(dict(coef=fit.params, se=fit.bse, p=fit.pvalues)).reset_index(names="term")
    table["n"] = int(fit.nobs)
    table["r2"] = fit.rsquared
    return table


def condition_table(df, last, compatibility_column):
    aggregations = dict(coupling=("coupling", "mean"), coupling_rate=("coupling_rate", "mean"))
    if compatibility_column:
        aggregations["compatibility"] = (compatibility_column, "mean")
    per_condition = df.groupby("condition").agg(**aggregations)
    final = last.groupby("condition").agg(final_acc=("test_acc", "mean"), final_acc_sd=("test_acc", "std"),
                                          final_loss=("test_loss", "mean"), runs=("network", "size"))
    return per_condition.join(final)


def lag_table(df):
    rows = []
    for k in (1, 2, 3):
        measures = (("euclidean", f"lag{k}_cos_current_b"), ("fisher", f"lag{k}_fisher_cos_current_b"),
                    ("fisher_gate_time", f"gate_lag{k}_fisher_cos"))
        for kind, column in measures:
            if column not in df or not df[column].notna().any():
                continue
            v = df[column].dropna()
            rows.append(dict(lag=k, measure=kind, column=column, n=len(v), mean=v.mean(),
                             median=v.median(), frac_negative=(v < 0).mean()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = L.load(*a.runs)
    df["condition"] = label(df)
    df["improve"] = -df.next_dloss
    next_acc = df.groupby(["network", "split"]).test_acc.shift(-1)
    df["acc_improve"] = next_acc - df.test_acc
    df["log_coupling"] = np.log1p(df.coupling)
    df["log_step"] = np.log(df.a_dot_norm.clip(lower=1e-12))
    compatibility = "update_fisher_cos_current_b" if "update_fisher_cos_current_b" in df else None
    lines = []

    last = df.sort_values("epoch").groupby("network").tail(1)
    conditions = condition_table(df, last, compatibility)
    conditions.reset_index().to_csv(out / "conditions.csv", index=False)
    lines += ["CONDITIONS", conditions.round(4).to_string(), ""]

    if "sgd" in set(df.condition):
        ref = "sgd"
    else:
        ref = sorted(df.condition.unique())[0]
    treatment = f"C(condition, Treatment('{ref}'))"
    coupling_model = clustered(f"log_coupling ~ {treatment} + progress + log_step", df)
    coupling_model.to_csv(out / "coupling_model.csv", index=False)
    lines += [f"COUPLING MODEL (reference = {ref}, clustered by run)",
              coupling_model.round(5).to_string(index=False), ""]

    per_run = local.per_run(df, ["coupling"], {"position": ["progress"]}, outcome="improve")
    per_net = local.by_network(per_run)
    local_rows = []
    for condition, g in per_net.groupby("condition"):
        local_rows.append(dict(condition=condition, **one_sample(g.rho)))
    local_rows.append(dict(condition="all", **one_sample(per_net.rho)))
    local_table = pd.DataFrame(local_rows)
    local_table.to_csv(out / "local.csv", index=False)
    lines += ["LOCAL: within-run rho(coupling, next-epoch improvement), position-controlled",
              local_table.round(4).to_string(index=False), ""]

    for outcome in ("improve", "acc_improve"):
        formula = f"{outcome} ~ log_coupling + {treatment} + progress + log_step + overfit_gap + resid_frac"
        next_model = clustered(formula, df)
        next_model.to_csv(out / f"next_model_{outcome}.csv", index=False)
        coupling_term = next_model[next_model.term == "log_coupling"]
        lines += [f"NEXT-EPOCH {outcome.upper()} MODEL (clustered)", coupling_term.round(6).to_string(index=False), ""]

    lags = lag_table(df)
    lags.to_csv(out / "lags.csv", index=False)
    lags_text = lags.round(4).to_string(index=False) if len(lags) else "(none)"
    lines += ["LAG COMPATIBILITY WITH CURRENT GRADIENT", lags_text, ""]

    accept_columns = [c for c in df if c.startswith("epoch_lag") and c.endswith("accept_rate")]
    if accept_columns:
        rates = df.groupby("condition")[accept_columns].mean()
        rates.reset_index().to_csv(out / "gates.csv", index=False)
        lines += ["GATE ACCEPTANCE RATES", rates.round(4).to_string(), ""]

    by_seed = last.pivot_table(index="seed", columns="condition", values="test_acc")
    paired_rows = []
    for condition in by_seed:
        if condition != ref and ref in by_seed:
            paired_rows.append(dict(condition=condition, baseline=ref, **paired(by_seed[condition], by_seed[ref])))
    paired_table = pd.DataFrame(paired_rows)
    paired_table.to_csv(out / "paired.csv", index=False)
    paired_text = paired_table.round(4).to_string(index=False) if len(paired_table) else "(none)"
    lines += [f"FINAL ACCURACY minus {ref}, paired by seed", paired_text]
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
