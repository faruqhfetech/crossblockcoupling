"""Loss change vs the uniform allocation, per intervention and horizon (unit = run).

  python -m analysis.allocation --input results/allocation --out results/report_allocation
"""
import argparse
from pathlib import Path

import pandas as pd

from .stats import one_sample, partial_rank_corr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(Path(a.input) / "allocation_interventions.csv")

    keys = ["run_id", "checkpoint_id", "horizon_steps"]
    uniform = d[d.intervention == "uniform"][keys + ["loss_change", "signed_change"]]
    paired_rows = d.merge(uniform, on=keys, suffixes=("", "_uniform"))
    paired_rows["loss_vs_uniform"] = paired_rows.loss_change - paired_rows.loss_change_uniform
    paired_rows["signed_vs_uniform"] = paired_rows.signed_change - paired_rows.signed_change_uniform
    paired_rows.to_csv(out / "paired.csv", index=False)

    per_run = paired_rows.groupby(["arch", "intervention", "horizon_steps", "run_id"], as_index=False)[
        ["loss_vs_uniform", "signed_vs_uniform"]].mean()
    rows = []
    for (arch, intervention, horizon), g in per_run.groupby(["arch", "intervention", "horizon_steps"]):
        row = dict(arch=arch, intervention=intervention, horizon_steps=horizon,
                   mean_signed_vs_uniform=g.signed_vs_uniform.mean())
        for key, value in one_sample(g.loss_vs_uniform).items():
            row[f"loss_{key}"] = value
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)

    corr_rows = []
    for (intervention, horizon), g in d.groupby(["intervention", "horizon_steps"]):
        rhos = []
        for _, run in g.groupby("run_id"):
            rhos.append(partial_rank_corr(run.signed_change, run.loss_change))
        corr_rows.append(dict(intervention=intervention, horizon_steps=horizon, **one_sample(rhos)))
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(out / "coupling_vs_loss.csv", index=False)

    text = ["LOSS vs UNIFORM by intervention and horizon (negative favors intervention; unit = run)",
            summary.round(5).to_string(index=False), "",
            "WITHIN-RUN rho(signed coupling, loss change)", corr.round(4).to_string(index=False)]
    (out / "summary.txt").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
