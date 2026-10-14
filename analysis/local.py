"""Within-run prediction of the next test-loss change; inference over networks."""
import numpy as np
import pandas as pd

from .stats import circular_null, one_sample, partial_rank_corr

CONTROLS = {
    "none": [],
    "position": ["progress"],
    "overfit_gap": ["progress", "overfit_gap"],
    "update_norm": ["progress", "update_norm"],
    "residual": ["progress", "resid_frac"],
    "persistence": ["progress", "a_dot_norm", "prev_dloss", "overfit_gap"],
}
PREDICTORS = ["coupling", "coupling_rate", "coupling_relative", "emp_delta_a", "reduction", "increase",
              "abs_H_a", "abs_D_a", "H_a", "D_a"]
SIGNED = {"coupling": "|delta_a|", "emp_delta_a": "signed delta_a", "reduction": "reduction max(-delta_a, 0)",
          "increase": "increase max(delta_a, 0)"}


def per_run(df, predictors=PREDICTORS, controls=CONTROLS, outcome="next_dloss"):
    """One partial rho per (network, split, predictor, control)."""
    extra = []
    if "probe_dloss_a" in df.columns and df["probe_dloss_a"].notna().any():
        extra = ["probe_dloss_a"]
    rows = []
    for (net, split), g in df.groupby(["network", "split"]):
        g = g.sort_values("epoch")
        condition = g["condition"].iloc[0] if "condition" in g else ""
        meta = dict(network=net, split=split, dataset=g.dataset.iloc[0], arch=g.arch.iloc[0],
                    condition=condition, n=len(g))
        for predictor in predictors:
            if predictor not in g or g[predictor].isna().all():
                continue
            for control_name, columns in controls.items():
                if control_name == "persistence":
                    columns = columns + extra
                if any(c not in g for c in columns):
                    continue
                Z = g[columns].to_numpy() if columns else None
                rho = partial_rank_corr(g[predictor], g[outcome], Z)
                rows.append(dict(meta, predictor=predictor, control=control_name, rho=rho))
    return pd.DataFrame(rows)


def by_network(per: pd.DataFrame) -> pd.DataFrame:
    """Average the splits of each network, so the network is the unit."""
    keys = ["network", "dataset", "arch", "condition", "predictor", "control"]
    return per.groupby(keys, dropna=False, as_index=False)["rho"].mean()


def summarize(per_net: pd.DataFrame, strata=("arch", "dataset")) -> pd.DataFrame:
    rows = []
    for (predictor, control), g in per_net.groupby(["predictor", "control"]):
        rows.append(dict(stratum="all", predictor=predictor, control=control, **one_sample(g.rho)))
        for stratum in strata:
            for value, h in g.groupby(stratum):
                rows.append(dict(stratum=f"{stratum}={value}", predictor=predictor, control=control,
                                 **one_sample(h.rho)))
    return pd.DataFrame(rows)


def table(per_net, predictors, controls=("position", "persistence"), by="arch"):
    """Mean rho, fraction negative and Wilcoxon p per predictor x control x group."""
    selected = per_net[per_net.predictor.isin(predictors) & per_net.control.isin(controls)]
    rows = []
    for (predictor, control), g in selected.groupby(["predictor", "control"]):
        groups = [("all", g)] + list(g.groupby(by))
        for group, h in groups:
            s = one_sample(h.rho)
            rows.append(dict(predictor=predictor, control=control, group=group, n=s["n"], mean=s["mean"],
                             frac_negative=s["frac_negative"], p_wilcoxon=s["p_wilcoxon"]))
    return pd.DataFrame(rows)


def decomposition_shares(df):
    """Median share of |H|, |D|, |I| in the exact split of delta_a (all-Linear upper blocks)."""
    if not {"H_a", "D_a", "I_a"}.issubset(df.columns):
        return pd.DataFrame()
    d = df.dropna(subset=["H_a", "D_a", "I_a"])
    if d.empty:
        return pd.DataFrame()
    total = d[["H_a", "D_a", "I_a"]].abs().sum(1)
    identity_error = (d.H_a + d.D_a + d.I_a - d.emp_delta_a).abs() / d.emp_delta_a.abs().clip(lower=1e-12)
    shares = d.assign(share_H=d.H_a.abs() / total, share_D=d.D_a.abs() / total, share_I=d.I_a.abs() / total,
                      identity_err=identity_error)
    columns = ["share_H", "share_D", "share_I", "identity_err"]
    return shares.groupby(["dataset", "split"])[columns].median().reset_index()


def circular_shift_test(df, predictor="coupling", n_perm=10000, seed=0):
    """Global two-sided test: mean observed within-network rho vs means of one
    random circular shift per (network, split), position-controlled."""
    rng = np.random.default_rng(seed)
    observed, null_banks = [], []
    for _, g in df.groupby(["network", "split"]):
        g = g.dropna(subset=[predictor, "next_dloss"]).sort_values("epoch")
        if len(g) < 5:
            continue
        Z = g[["progress"]].to_numpy()
        rho = partial_rank_corr(g[predictor], g.next_dloss, Z)
        bank = circular_null(g[predictor].to_numpy(), g.next_dloss.to_numpy(), Z)
        bank = bank[np.isfinite(bank)]
        if np.isfinite(rho) and bank.size:
            observed.append(rho)
            null_banks.append(bank)

    if not observed:
        return dict(predictor=predictor, n=0, observed_mean=np.nan, null_mean=np.nan,
                    null_q025=np.nan, null_q975=np.nan, p_two_sided=np.nan)
    observed = np.array(observed)

    null_means = []
    for _ in range(n_perm):
        one_shift_each = [bank[rng.integers(len(bank))] for bank in null_banks]
        null_means.append(np.mean(one_shift_each))
    null_means = np.array(null_means)

    observed_mean = observed.mean()
    center = null_means.mean()
    as_extreme = np.sum(np.abs(null_means - center) >= abs(observed_mean - center))
    p = (1 + as_extreme) / (n_perm + 1)
    return dict(predictor=predictor, n=len(observed), observed_mean=observed_mean, null_mean=center,
                null_q025=np.quantile(null_means, .025), null_q975=np.quantile(null_means, .975), p_two_sided=p)


def phases(df, predictor="coupling"):
    """Raw within-run Spearman in equal-count thirds of each run."""
    rows = []
    for (net, split), g in df.groupby(["network", "split"]):
        g = g.sort_values("epoch")
        thirds = np.array_split(np.arange(len(g)), 3)
        for name, idx in zip(("early", "middle", "late"), thirds):
            part = g.iloc[idx]
            rho = partial_rank_corr(part[predictor], part.next_dloss, None, min_n=3)
            rows.append(dict(network=net, split=split, phase=name, rho=rho))
    per_net = pd.DataFrame(rows).groupby(["network", "phase"], as_index=False).rho.mean()
    return pd.DataFrame([dict(phase=phase, **one_sample(g.rho)) for phase, g in per_net.groupby("phase")])
