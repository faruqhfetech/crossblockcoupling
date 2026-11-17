"""Coupling vs the next-epoch test-loss change in all eight settings, under every control.

Settings: MLPs on MNIST and CIFAR-10 (stage 1), GroupNorm CNNs with and without skip connections on
CIFAR-10 (stage 1) and MNIST (stage 2), and BatchNorm CNNs on CIFAR-10 (stage 3).
Unit = network (block splits averaged); negative rho = coupling comes before improvement.

  python -m analysis.settings_grid          # writes results/final/settings_grid.csv, .txt and _per_network.csv
"""
from pathlib import Path

import pandas as pd

from . import load as L
from . import local
from .stats import one_sample

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "final"
SETTINGS = [
    ("MLP, MNIST", "base_obs_mnist", None),
    ("MLP, CIFAR-10", "base_obs_cifar10", None),
    ("CNN, MNIST", "cnn_mnist", False),
    ("CNN + skip, MNIST", "cnn_mnist", True),
    ("CNN, CIFAR-10", "base_cnn_cifar10", False),
    ("CNN + skip, CIFAR-10", "base_cnn_cifar10", True),
    ("CNN BN, CIFAR-10", "cnn_bn", False),
    ("CNN BN + skip, CIFAR-10", "cnn_bn", True),
]
CONTROLS = {
    "position": ["progress"],
    "+ residual": ["progress", "resid_frac"],
    "+ update norm": ["progress", "update_norm"],
    "+ train-test gap": ["progress", "overfit_gap"],
    "+ Fisher trace level": ["progress", "emp_trace_b"],
    "+ norm product": ["progress", "rms_product"],
    "+ persistence": ["progress", "a_dot_norm", "prev_dloss", "overfit_gap"],
    "none": [],
}


def setting_frame(battery, skip):
    df, _ = L.healthy(L.load(RESULTS / battery))
    if skip is not None:
        df = df[df.skip == skip]
    return df


def cell(rhos):
    s = one_sample(rhos)
    negative = int(round(s["frac_negative"] * s["n"]))
    return dict(mean=round(s["mean"], 2), negative=f"{negative}/{s['n']}", p_wilcoxon=round(s["p_wilcoxon"], 4))


def main():
    rows = []
    per_network = []
    for label, battery, skip in SETTINGS:
        df = setting_frame(battery, skip)
        available = {name: cols for name, cols in CONTROLS.items() if all(c in df for c in cols)
                     and not any(df[c].isna().all() for c in cols)}
        renamed = {("persistence" if name == "+ persistence" else name): cols for name, cols in available.items()}
        per = local.per_run(df, predictors=["coupling"], controls=renamed)
        per["control"] = per.control.replace({"persistence": "+ persistence"})
        net = local.by_network(per)
        per_network.append(net.assign(setting=label))
        for control in CONTROLS:
            values = net[net.control == control].rho
            if values.empty:
                rows.append(dict(setting=label, control=control, mean=None, negative="n/a", p_wilcoxon=None))
                continue
            rows.append(dict(setting=label, control=control, **cell(values)))
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "settings_grid.csv", index=False)
    pd.concat(per_network, ignore_index=True).to_csv(OUT / "settings_grid_per_network.csv", index=False)
    wide = table.assign(text=table.apply(lambda r: "n/a" if r["mean"] is None or pd.isna(r["mean"])
                                         else f"{r['mean']:+.2f} {r['negative']} p={r['p_wilcoxon']:.3f}", axis=1))
    text = wide.pivot(index="control", columns="setting", values="text").reindex(list(CONTROLS))
    text = text[[label for label, _, _ in SETTINGS]]
    (OUT / "settings_grid.txt").write_text(text.to_string(), encoding="utf-8")
    print(text.to_string())


if __name__ == "__main__":
    main()
