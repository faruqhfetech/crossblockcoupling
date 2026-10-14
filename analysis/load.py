"""Load run directories into one frame keyed by `network` (one trained network)."""
from dataclasses import fields

import numpy as np
import pandas as pd

from cbfc.config import RunConfig
from cbfc.io import load_runs

NOT_CONDITION = {"depth", "seed", "save_checkpoints", "tag", "dataset", "arch", "splits", "extra_probes",
                 "diag_fisher", "predictors", "save_matrices", "model_fisher", "probe_source", "probe_size", "probe_seed"}


def condition_labels(df, varying):
    """'key=value+key=value' over the varying config fields, one label per row."""
    def label(row):
        return "+".join(f"{key}={value}" for key, value in row.items())
    return df[varying].astype(str).apply(label, axis=1)


def load(*dirs):
    """`condition` = the config fields that vary across the loaded runs."""
    df = load_runs(*dirs)
    if df.empty:
        raise SystemExit(f"no runs found in {dirs}")
    varying = []
    for f in fields(RunConfig):
        if f.name in NOT_CONDITION or f.name not in df:
            continue
        if df[f.name].astype(str).nunique() > 1:
            varying.append(f.name)
    if varying:
        df["condition"] = condition_labels(df, varying)
    else:
        df["condition"] = "base"
    df["network"] = df.run_id
    return add_derived(df)


def add_derived(df):
    df = df.sort_values(["network", "split", "epoch"]).copy()
    series = df.groupby(["network", "split"], sort=False)
    df["next_dloss"] = series.test_loss.shift(-1) - df.test_loss
    df["prev_dloss"] = df.test_loss - series.test_loss.shift(1)
    last_index = series.epoch.transform("size").sub(1).clip(lower=1)
    df["progress"] = series.cumcount() / last_index

    df["coupling"] = df.emp_delta_a.abs()
    df["reduction"] = (-df.emp_delta_a).clip(lower=0)
    df["increase"] = df.emp_delta_a.clip(lower=0)
    for part in ("H_a", "D_a", "I_a"):
        if part in df:
            df[f"abs_{part}"] = df[part].abs()

    df["coupling_rate"] = df.coupling / df.a_dot_norm.replace(0, np.nan)
    df["coupling_relative"] = df.coupling / df.emp_trace_b.abs().replace(0, np.nan)
    df["overfit_gap"] = df.test_loss - df.train_loss_running
    df["update_norm"] = np.hypot(df.a_dot_norm, df.b_dot_norm)
    total_change = df.emp_delta_a.abs() + df.emp_delta_b.abs()
    df["resid_frac"] = df.emp_residual.abs() / total_change.replace(0, np.nan)
    return df


def has_spike(values, spike_ratio):
    """Max |value| over spike_ratio x the median |value| (needs 3+ values)."""
    v = values.abs().dropna()
    if v.size < 3:
        return False
    return v.max() > spike_ratio * max(v.median(), 1e-6)


def health(df, chance_margin=0.05, spike_ratio=50.0):
    """training_failed: final test acc near chance. unstable: coupling per unit motion or
    tr F_bb exceeds spike_ratio x its run median."""
    rows = []
    for net, g in df.groupby("network"):
        g = g.sort_values("epoch")
        n_classes = 100 if g.dataset.iloc[0] == "cifar100" else 10
        chance = 1 / n_classes
        final_acc = g.test_acc.iloc[-1]
        unstable = has_spike(g["coupling_rate"], spike_ratio) or has_spike(g["emp_trace_b"], spike_ratio)
        rows.append(dict(network=net, final_test_acc=final_acc,
                         training_failed=bool(final_acc < chance + chance_margin), unstable=unstable))
    return pd.DataFrame(rows)


def healthy(df):
    h = health(df)
    good = h.network[~(h.training_failed | h.unstable)]
    return df[df.network.isin(good)], h
