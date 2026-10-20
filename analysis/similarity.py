"""Similarity / norm baselines vs coupling, persistence controls, H/D/I composition, exact vs diagonal Fisher.

  python -m analysis.similarity --runs results/sim_cifar10 --out results/report_sim [--probe train]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import load as L
from .stats import one_sample, partial_rank_corr

BASELINES = ["cka_adj", "cos_adj", "cka_boundary", "rms_b_input", "bcka_adj", "brms_b_input",
             "rms_product", "jac_adj", "jac_boundary", "jb_fro"]
CTRL_SETS = [("pos", ["progress"]), ("pos+step", ["progress", "a_dot_norm"]),
             ("pos+dL_a", ["progress", "probe_dloss_a"]),
             ("pos+dL_a+prev", ["progress", "probe_dloss_a", "prev_dloss"]),
             ("all", ["progress", "a_dot_norm", "probe_dloss_a", "prev_dloss", "overfit_gap"])]


def within(df, x, y, controls):
    """Per condition: summary of within-network partial rho(x, y | controls)."""
    rows = []
    for (condition, _), g in df.groupby(["condition", "network"]):
        g = g.sort_values("epoch")
        if x not in g or y not in g or g[x].isna().all():
            continue
        rho = partial_rank_corr(g[x], g[y], g[controls].to_numpy())
        rows.append(dict(condition=condition, rho=rho))
    per_network = pd.DataFrame(rows, columns=["condition", "rho"])
    out = []
    for condition, g in per_network.groupby("condition"):
        out.append(dict(condition=condition, x=x, y=y, controls="+".join(controls), **one_sample(g.rho)))
    return out


def use_probe(df, probe):
    """Swap *_<probe> columns into the unsuffixed names."""
    if not probe:
        return df
    suffix = f"_{probe}"
    suffixed = [c for c in df.columns if c.endswith(suffix)]
    if not suffixed:
        raise SystemExit(f"no *{suffix} columns")
    renames = {c: c[:-len(suffix)] for c in suffixed}
    replaced = [plain for plain in renames.values() if plain in df]
    return L.add_derived(df.drop(columns=replaced).rename(columns=renames))


def decomposition_table(df):
    total = df[["H_a", "D_a", "I_a"]].abs().sum(1)
    identity_error = (df.H_a + df.D_a + df.I_a - df.emp_delta_a).abs() / df.emp_delta_a.abs().clip(lower=1e-12)
    shares = df.assign(share_H=df.H_a.abs() / total, share_D=df.D_a.abs() / total,
                       share_I=df.I_a.abs() / total, identity_err=identity_error)
    table = shares.groupby("condition")[["share_H", "share_D", "share_I", "identity_err"]].median()

    layer_columns = sorted(c for c in df if c.startswith("D_a_L") and c[5:].isdigit())
    layer_total = df[layer_columns].abs().sum(1)
    for column in layer_columns:
        share = df[column].abs() / layer_total
        table[f"share_{column}"] = share.groupby(df.condition).median()
    right_wrong_total = df.D_a_right.abs() + df.D_a_wrong.abs()
    for column in ("D_a_right", "D_a_wrong"):
        share = df[column].abs() / right_wrong_total
        table[f"share_{column}"] = share.groupby(df.condition).median()
    return table


def diagonal_table(df):
    rows = []
    for condition, g in df.groupby("condition"):
        ratio = g.diag_trace_b / g.emp_trace_b
        level_rhos = []
        coupling_rhos = []
        for _, network in g.groupby("network"):
            level_rhos.append(partial_rank_corr(network.emp_trace_b, network.diag_trace_b))
            coupling_rhos.append(partial_rank_corr(network.emp_delta_a.abs(), network.diag_delta_a.abs()))
        sign_agreement = (np.sign(g.emp_delta_a) == np.sign(g.diag_delta_a)).mean()
        magnitude_ratio = (g.diag_delta_a.abs() / g.emp_delta_a.abs()).median()
        rows.append(dict(condition=condition, level_ratio_median=ratio.median(),
                         level_rho_mean=np.nanmean(level_rhos), coupling_rho_mean=np.nanmean(coupling_rhos),
                         coupling_sign_agreement=sign_agreement, coupling_mag_ratio_median=magnitude_ratio))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe", default="", help="extra probe suffix to analyze, e.g. train")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = use_probe(L.load(*a.runs), a.probe)
    fab_norm = np.sqrt(df.emp_Fab_frob_sq.clip(lower=0))
    df["Fab_norm"] = fab_norm / np.sqrt(df.emp_trace_a * df.emp_trace_b)
    for k in BASELINES:
        if f"d_{k}_a" in df:
            df[f"abs_d_{k}_a"] = df[f"d_{k}_a"].abs()
    present = [k for k in BASELINES if k in df and df[k].notna().any()]
    lines = []

    columns = ["test_acc", "coupling", "coupling_rate", "Fab_norm"] + present
    run_means = df.groupby(["condition", "network"])[columns].mean()
    means = run_means.groupby("condition").agg(["mean", "std"])
    means.to_csv(out / "1_run_means.csv")
    lines += ["1. RUN-LEVEL MEANS", means.T.round(4).to_string(), ""]

    rows = []
    for k in present:
        for controls in (["progress"], ["progress", "a_dot_norm"]):
            rows += within(df, k, "coupling", controls)
            if f"abs_d_{k}_a" in df:
                rows += within(df, f"abs_d_{k}_a", "coupling", controls)
    rows += within(df, "a_dot_norm", "coupling", ["progress"])
    vs_coupling = pd.DataFrame(rows)
    vs_coupling.to_csv(out / "2_baselines_vs_coupling.csv", index=False)
    lines += ["2. BASELINES vs COUPLING (mean within-run rho)",
              vs_coupling.pivot_table(index="x", columns=["condition", "controls"], values="mean").round(3).to_string(),
              ""]

    predictors = ["coupling", "emp_delta_a", "D_a", "H_a", "probe_dloss_a", "prev_dloss"] + present
    rows = []
    for predictor in predictors:
        for set_name, controls in CTRL_SETS:
            if predictor in controls or any(c not in df for c in controls):
                continue
            for result in within(df, predictor, "next_dloss", controls):
                rows.append(dict(result, control_set=set_name))
    persistence = pd.DataFrame(rows)
    persistence.to_csv(out / "3_persistence.csv", index=False)
    if len(persistence):
        table = persistence.pivot_table(index="x", columns=["condition", "control_set"], values="mean")
        lines += ["3. NEXT-INTERVAL LOSS PREDICTION under persistence controls (negative = improvement)",
                  table.round(3).to_string(), ""]

    if "H_a" in df:
        decomposition = decomposition_table(df)
        decomposition.to_csv(out / "4_decomposition.csv")
        lines += ["4. H/D/I COMPOSITION (medians)", decomposition.round(3).T.to_string(), ""]

    if "diag_trace_b" in df:
        diagonal = diagonal_table(df)
        diagonal.to_csv(out / "5_diagonal_vs_exact.csv", index=False)
        lines += ["5. EXACT vs DIAGONAL FISHER", diagonal.round(3).to_string(index=False)]

    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
