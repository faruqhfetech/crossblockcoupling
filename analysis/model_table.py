"""Model description table for the final report: parameter counts per architecture, lower and upper block
sizes at each split, and output shapes through the network.

  python -m analysis.model_table              # writes results/final/model_table.txt
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "final" / "model_table.txt"
BATTERIES = ["base_obs_mnist", "base_obs_cifar10", "base_cnn_cifar10", "cnn_bn", "sim_cifar10",
             "cnn_fisher_check", "momentum_history", "vit_cifar100"]


def first_rows(battery):
    rows = []
    for path in sorted((RESULTS / battery).glob("*.csv")):
        if path.name.endswith("_matrices.csv"):
            continue
        df = pd.read_csv(path, nrows=4)
        rows.append(df.iloc[:2])
    return pd.concat(rows, ignore_index=True)


def main():
    lines = []
    for battery in BATTERIES:
        df = first_rows(battery)
        df["total"] = df.n_params_a + df.n_params_b
        keep = [c for c in ("condition", "split", "split_cut", "n_params_a", "n_params_b", "total") if c in df]
        table = df[keep].drop_duplicates().sort_values(["condition", "split"])
        lines += [f"== {battery}", table.to_string(index=False), ""]
    text = "\n".join(lines)
    OUT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
