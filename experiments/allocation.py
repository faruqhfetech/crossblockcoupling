"""Update allocation: at each checkpoint rescale one real update to (s_a delta_a, s_b delta_b),
optionally back to ||delta||, and measure the lower-induced tr F_bb change, the immediate
loss, and the loss after h steps in forked branches (paired batches across interventions).

  python -m experiments.allocation --dataset cifar10 --archs mlp rescnn --depths 7 15
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from cbfc import data as D
from cbfc import env, optim
from cbfc.config import RunConfig
from cbfc.fisher import block_stats
from cbfc.io import write_rows
from cbfc.split import make_split
from cbfc.train import build_model, get_data
from experiments.validation import norm


def interventions(a):
    """{name: (lower_scale, upper_scale)} for the chosen and custom interventions."""
    known = {"uniform": (1, 1), "lower_hot": (a.hot, a.cold), "upper_hot": (a.cold, a.hot),
             "lower_only": (1, 0), "upper_only": (0, 1), "both_hot": (a.hot, a.hot), "both_cold": (a.cold, a.cold)}
    chosen = {name: known[name] for name in a.interventions}
    for spec in a.custom:
        name, scale_a, scale_b = spec.split(":")
        chosen[name] = (float(scale_a), float(scale_b))
    return chosen


def scaled_update(delta, names_a, scale_a, scale_b, match_norm, all_names):
    scaled = {}
    for name, value in delta.items():
        scale = scale_a if name in names_a else scale_b
        scaled[name] = value * scale
    if match_norm:
        factor = norm(delta, all_names) / max(norm(scaled, all_names), 1e-30)
        scaled = {name: value * factor for name, value in scaled.items()}
    return scaled


def run_one(cfg, a, data, out, rows, checkpoint_rows):
    torch.manual_seed(cfg.seed)
    model = build_model(cfg, data).to(data.train.y.device)
    split = make_split(model, cfg.splits[0])
    opt = optim.build(model, "sgd", cfg.lr, cfg.momentum)
    probe_x, probe_y, _ = D.probe(data, cfg.probe_source, cfg.probe_size, cfg.probe_seed)
    eval_split = D.Split(data.val.x[:a.eval_size], data.val.y[:a.eval_size])
    gen = torch.Generator().manual_seed(cfg.seed)
    stream = D.Stream(data.train, cfg.batch_size, cfg.seed + 1_000_000)
    branch_stream = D.Stream(data.train, cfg.batch_size, cfg.seed + 2_000_000)
    rid = cfg.run_id()
    all_names = split.names_a + split.names_b
    checkpoint = 0

    def trace_b(params, buffers):
        stats = block_stats(model, params, buffers, probe_x, probe_y, split.names_a, split.names_b,
                            gram=False, model_fisher=False, decomposition=False)
        return stats["emp_trace_b"]

    for epoch in range(1, cfg.epochs + 1):
        for x, y in D.batches(data.train, cfg.batch_size, gen):
            optim.step_delta(model, opt, x, y)
        if epoch % cfg.checkpoint_every:
            continue
        checkpoint += 1
        base = {name: p.detach().clone() for name, p in model.named_parameters()}
        buffers = dict(model.named_buffers())
        loss0, acc0 = D.evaluate(model, eval_split)
        checkpoint_rows.append(dict(run_id=rid, arch=cfg.arch, depth=cfg.depth, seed=cfg.seed, epoch=epoch,
                                    checkpoint_id=checkpoint, eval_loss=loss0, eval_acc=acc0))

        x, y = stream.next()
        model_copy, opt_copy = optim.fork(model, opt)
        delta = optim.step_delta(model_copy, opt_copy, x, y)
        branch_batches = [branch_stream.next() for _ in range(max(a.horizons))]
        trace0 = trace_b(base, buffers)

        for name, (scale_a, scale_b) in interventions(a).items():
            update = scaled_update(delta, split.names_a, scale_a, scale_b, a.match_norm, all_names)
            moved = {n: base[n] + update[n] for n in base}
            lower_only = {n: moved[n] if n in split.names_a else base[n] for n in base}
            trace1 = trace_b(lower_only, buffers)
            common = dict(run_id=rid, arch=cfg.arch, depth=cfg.depth, seed=cfg.seed, epoch=epoch,
                          checkpoint_id=checkpoint, intervention=name, lower_scale=scale_a, upper_scale=scale_b,
                          norm_matched=a.match_norm, update_norm=norm(update, all_names),
                          lower_update_norm=norm(update, split.names_a), upper_update_norm=norm(update, split.names_b),
                          trace_b_before=trace0, trace_b_after_lower=trace1, signed_change=trace1 - trace0,
                          absolute_change=abs(trace1 - trace0), loss_before=loss0)

            for horizon in sorted(set(a.horizons)):
                if horizon == 0:
                    loss_h, acc_h = D.evaluate(model, eval_split, moved, buffers)
                else:
                    branch_model, branch_opt = optim.fork(model, opt)
                    with torch.no_grad():
                        for n, p in branch_model.named_parameters():
                            p.copy_(moved[n])
                    for bx, by in branch_batches[:horizon]:
                        optim.step_delta(branch_model, branch_opt, bx, by)
                    loss_h, acc_h = D.evaluate(branch_model, eval_split)
                rows.append(dict(common, horizon_steps=horizon, loss_after=loss_h, loss_change=loss_h - loss0,
                                 acc_after=acc_h))

        optim.step_delta(model, opt, x, y)
        write_rows(out / "allocation_interventions.csv", rows)
        write_rows(out / "allocation_checkpoints.csv", checkpoint_rows)
        print(f"[{rid}] epoch {epoch} checkpoint {checkpoint} eval_acc {acc0:.4f}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--fake", action="store_true")
    p.add_argument("--archs", nargs="+", default=["mlp", "rescnn"])
    p.add_argument("--depths", nargs="+", type=int, default=[7, 15])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--mlp-width", type=int, default=128)
    p.add_argument("--cnn-width", type=int, default=64)
    p.add_argument("--skip", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--split", default="depth_fraction")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--interventions", nargs="+", default=["uniform", "lower_hot", "upper_hot", "lower_only",
                                                           "upper_only", "both_hot", "both_cold"])
    p.add_argument("--custom", nargs="*", default=[], help="NAME:S_A:S_B")
    p.add_argument("--hot", type=float, default=2.0)
    p.add_argument("--cold", type=float, default=0.5)
    p.add_argument("--match-norm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--horizons", nargs="+", type=int, default=[0, 1, 10, 100])
    p.add_argument("--eval-size", type=int, default=2000)
    p.add_argument("--probe-size", type=int, default=200)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    out = Path(a.out) if a.out else env.results_root() / "allocation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "allocation_config.json").write_text(json.dumps(vars(a), indent=2))
    rows, checkpoint_rows = [], []
    for arch in a.archs:
        for depth in a.depths:
            for seed in a.seeds:
                width = a.mlp_width if arch == "mlp" else a.cnn_width
                val_size = 256 if a.fake else 5000
                cfg = RunConfig(dataset=a.dataset, fake=a.fake, val_size=val_size, arch=arch, depth=depth,
                                seed=seed, skip=a.skip, width=width, splits=(a.split,), lr=a.lr,
                                momentum=a.momentum, epochs=a.epochs, checkpoint_every=a.checkpoint_every,
                                probe_size=a.probe_size, tag="alloc")
                data = get_data(cfg, a.device or env.device())
                run_one(cfg, a, data, out, rows, checkpoint_rows)


if __name__ == "__main__":
    main()
