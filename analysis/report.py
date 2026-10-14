"""Headline analysis.

  python -m analysis.report --runs results/obs_cifar10 --out results/report_obs
  python -m analysis.report --runs results/sched_cifar10 --out results/report_sched       --paired schedule=crossfade schedule=none schedule=cosine schedule=linear
"""
import argparse
from pathlib import Path

import pandas as pd

from . import endpoints as E
from . import load as L
from . import local


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-unhealthy", action="store_true")
    ap.add_argument("--paired", nargs="+", metavar="COND",
                    help="TREATMENT BASELINE [BASELINE ...]: paired endpoint tests")
    ap.add_argument("--permutations", type=int, default=10000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = L.load(*a.runs)
    clean, health = L.healthy(df)
    health.to_csv(out / "health.csv", index=False)
    if a.include_unhealthy:
        clean = df
    if clean.empty:
        raise SystemExit("no healthy networks (all at chance or unstable); see health.csv "
                         "or rerun with --include-unhealthy")

    flagged = int((health.training_failed | health.unstable).sum())
    health_line = f"networks: {df.network.nunique()} loaded, {len(health) - flagged} healthy ({flagged} flagged)"
    if a.include_unhealthy:
        health_line += ", all included"
    n_series = clean.groupby(["network", "split"]).ngroups
    lines = [health_line, f"(network, split) series: {n_series}", ""]

    per = local.per_run(clean)
    per.to_csv(out / "local_per_run.csv", index=False)
    per_net = local.by_network(per)
    summary = local.summarize(per_net)
    summary.to_csv(out / "local_summary.csv", index=False)
    headline = summary[summary.predictor == "coupling"]
    lines.append("LOCAL PREDICTION: coupling -> next test-loss change (unit = network; negative = improvement)")
    columns = ["stratum", "control", "n", "mean", "median", "frac_negative", "p_wilcoxon"]
    lines.append(headline[columns].round(4).to_string(index=False))

    signed = local.table(per_net, list(local.SIGNED))
    signed["predictor"] = signed.predictor.map(local.SIGNED)
    signed.to_csv(out / "signed_vs_unsigned.csv", index=False)
    lines += ["", "SIGNED vs UNSIGNED (positive rho for signed delta_a = reductions precede improvement)",
              signed.round(4).to_string(index=False)]

    shares = local.decomposition_shares(clean)
    if len(shares):
        hd = local.table(per_net, ["abs_H_a", "abs_D_a", "H_a", "D_a"], by="dataset")
        shares.to_csv(out / "decomposition_shares.csv", index=False)
        hd.to_csv(out / "decomposition_prediction.csv", index=False)
        lines += ["", "INPUT/ERROR DECOMPOSITION delta_a = H + D + I (all-Linear upper blocks)",
                  shares.round(4).to_string(index=False), "",
                  "H (inputs moved) and D (errors moved) -> next test-loss change",
                  hd.round(4).to_string(index=False)]

    circular_rows = []
    for predictor in ("coupling", "coupling_rate", "emp_delta_a", "reduction"):
        circular_rows.append(local.circular_shift_test(clean, predictor, a.permutations))
    circular = pd.DataFrame(circular_rows)
    circular.to_csv(out / "circular_shift.csv", index=False)
    lines += ["", "CIRCULAR-SHIFT NULL (position-controlled, two-sided)", circular.round(4).to_string(index=False)]

    phase_table = local.phases(clean)
    phase_table.to_csv(out / "phases.csv", index=False)
    lines += ["", "PHASES (raw within-run rho, equal thirds)", phase_table.round(4).to_string(index=False)]

    ep = E.endpoints(clean)
    ep.to_csv(out / "endpoints.csv", index=False)
    concordance = E.concordance(ep)
    concordance.to_csv(out / "concordance.csv", index=False)
    if len(concordance):
        lines += ["", "FINAL ACCURACY: fixed-depth seed-pair concordance with mean coupling",
                  concordance.round(4).to_string(index=False)]

    if a.paired:
        outcomes = ["final_test_acc"]
        if "selected_test_acc" in ep:
            outcomes.append("selected_test_acc")
        treatment, baselines = a.paired[0], a.paired[1:]
        tests = [E.paired_conditions(ep, treatment, baselines, outcome) for outcome in outcomes]
        paired_table = pd.concat(tests)
        paired_table.to_csv(out / "paired.csv", index=False)
        lines += ["", "PAIRED ENDPOINTS", paired_table.round(4).to_string(index=False)]

    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
