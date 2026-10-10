"""Measurement validation: at each checkpoint, tr F_bb(a_t + alpha v, b_t) for the actual
lower update v and controls (opposite, norm-matched random, permuted, other_batch,
random_labels), over every probe source x seed x size x balance, plus the test-loss
change at several horizons on the real trajectory.

  python -m experiments.validation --dataset cifar10 --archs mlp rescnn --depths 7 15
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from cbfc import data as D
from cbfc import env, optim
from cbfc.config import RunConfig
from cbfc.fisher import block_stats
from cbfc.io import write_rows
from cbfc.split import make_split
from cbfc.train import build_model, get_data

TABLE_NAMES = ("measurements", "horizons", "checkpoints")


def norm(v, names):
    total = 0.0
    for name in names:
        total += float(v[name].double().pow(2).sum())
    return math.sqrt(total)


def random_like(ref, names, gen):
    """Random direction on the given names, rescaled to the norm of ref on those names."""
    out = {name: torch.zeros_like(t) for name, t in ref.items()}
    for name in names:
        out[name] = torch.randn(ref[name].shape, generator=gen).to(ref[name])
    scale = norm(ref, names) / max(norm(out, names), 1e-30)
    return {name: t * scale for name, t in out.items()}


def permuted(ref, names, gen):
    """The entries of ref on the given names, shuffled across those names."""
    flat = torch.cat([ref[name].reshape(-1) for name in names])
    order = torch.randperm(flat.numel(), generator=gen).to(flat.device)
    flat = flat[order]
    out = {name: torch.zeros_like(t) for name, t in ref.items()}
    start = 0
    for name in names:
        size = ref[name].numel()
        out[name] = flat[start:start + size].reshape_as(ref[name])
        start += size
    return out


def trace_b(model, params, buffers, x, y, split):
    stats = block_stats(model, params, buffers, x, y, split.names_a, split.names_b,
                        gram=False, model_fisher=False, decomposition=False)
    return stats["emp_trace_b"], stats["loss"].mean().item()


def move_lower(before, direction, alpha, names_a):
    """Parameters with block a moved by alpha * direction and block b unchanged."""
    moved = {}
    for name, value in before.items():
        if name in names_a:
            moved[name] = value + alpha * direction[name]
        else:
            moved[name] = value
    return moved


def make_probes(data, a):
    if a.class_balanced == "both":
        balance_options = [False, True]
    else:
        balance_options = [a.class_balanced == "yes"]
    probes = {}
    for source in a.probe_sources:
        for probe_seed in a.probe_seeds:
            for size in a.probe_sizes:
                for balanced in balance_options:
                    x, y, _ = D.probe(data, source, size, probe_seed, balanced)
                    probes[(source, probe_seed, size, balanced)] = (x, y)
    return probes


def run_one(cfg, a, data, out, tables):
    torch.manual_seed(cfg.seed)
    model = build_model(cfg, data).to(data.train.y.device)
    split = make_split(model, cfg.splits[0])
    opt = optim.build(model, "sgd", cfg.lr, cfg.momentum)
    rid = cfg.run_id()
    measurements, horizon_rows, checkpoint_rows = tables
    probes = make_probes(data, a)
    gen = torch.Generator().manual_seed(cfg.seed)
    stream = D.Stream(data.train, cfg.batch_size, cfg.seed + 1_000_000)
    steps_per_epoch = math.ceil(len(data.train) / cfg.batch_size)
    pending = []
    state = dict(step=0)
    meta = dict(run_id=rid, arch=cfg.arch, dataset=cfg.dataset, depth=cfg.depth, seed=cfg.seed)

    def advance(x, y):
        """One training step; record every horizon that this step reaches."""
        optim.step_delta(model, opt, x, y)
        state["step"] += 1
        due = [p for p in pending if p["target_step"] <= state["step"]]
        if not due:
            return
        loss, acc = D.evaluate(model, data.test)
        for p in due:
            horizon_rows.append(dict(p, horizon_steps=state["step"] - p["checkpoint_step"],
                                     test_loss_at_horizon=loss, test_acc_at_horizon=acc,
                                     test_loss_change=loss - p["test_loss_at_checkpoint"]))
        still_pending = [p for p in pending if p["target_step"] > state["step"]]
        pending[:] = still_pending

    def save():
        for name, rows in zip(TABLE_NAMES, tables):
            write_rows(out / f"validation_{name}.csv", rows)

    checkpoint = 0
    for epoch in range(1, cfg.epochs + 1):
        for x, y in D.batches(data.train, cfg.batch_size, gen):
            advance(x, y)
        if epoch % cfg.checkpoint_every:
            continue
        checkpoint += 1
        loss0, acc0 = D.evaluate(model, data.test)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        buffers = dict(model.named_buffers())
        actual_x, actual_y = stream.next()

        directions = {}
        if "other_batch" in a.directions:
            other_x, other_y = stream.next()
            model_copy, opt_copy = optim.fork(model, opt)
            directions["other_batch"] = optim.step_delta(model_copy, opt_copy, other_x, other_y)
        if "random_labels" in a.directions:
            label_gen = torch.Generator().manual_seed(cfg.seed * 7919 + checkpoint)
            random_y = torch.randint(data.classes, actual_y.shape, generator=label_gen).to(actual_y)
            model_copy, opt_copy = optim.fork(model, opt)
            directions["random_labels"] = optim.step_delta(model_copy, opt_copy, actual_x, random_y)
        actual = optim.step_delta(model, opt, actual_x, actual_y)
        directions["actual"] = actual
        state["step"] += 1

        direction_gen = torch.Generator().manual_seed(cfg.seed * 10_000 + checkpoint)
        if "opposite" in a.directions:
            directions["opposite"] = {name: -v for name, v in actual.items()}
        if "permuted" in a.directions:
            directions["permuted"] = permuted(actual, split.names_a, direction_gen)
        if "random" in a.directions:
            for r in range(a.random_directions):
                directions[f"random#{r}"] = random_like(actual, split.names_a, direction_gen)

        checkpoint_rows.append(dict(meta, epoch=epoch, step=state["step"], checkpoint_id=checkpoint,
                                    test_loss=loss0, test_acc=acc0,
                                    actual_a_update_norm=norm(actual, split.names_a),
                                    actual_b_update_norm=norm(actual, split.names_b)))

        primary_change = float("nan")
        selected = not a.measure_checkpoints or checkpoint in a.measure_checkpoints
        if selected and checkpoint % a.measure_every == 0:
            for probe_index, (key, (px, py)) in enumerate(probes.items()):
                source, probe_seed, size, balanced = key
                trace0, probe_loss0 = trace_b(model, before, buffers, px, py, split)
                for direction_name, direction in directions.items():
                    if "#" in direction_name:
                        kind, rep = direction_name.split("#")
                        rep = int(rep)
                    else:
                        kind, rep = direction_name, 0
                    direction_norm = norm(direction, split.names_a)
                    for alpha in a.alphas:
                        if alpha:
                            moved = move_lower(before, direction, alpha, split.names_a)
                            trace1, probe_loss1 = trace_b(model, moved, buffers, px, py, split)
                        else:
                            trace1, probe_loss1 = trace0, probe_loss0
                        change = trace1 - trace0
                        measurements.append(dict(
                            meta, epoch=epoch, checkpoint_id=checkpoint, probe_source=source,
                            probe_seed=probe_seed, probe_size=size, class_balanced=balanced,
                            direction=kind, direction_rep=rep, alpha=alpha, direction_norm=direction_norm,
                            trace_b_before=trace0, trace_b_cf=trace1, signed_change=change,
                            absolute_change=abs(change), relative_change=abs(change) / max(abs(trace0), 1e-12),
                            probe_loss_before=probe_loss0, probe_loss_cf=probe_loss1,
                            immediate_loss_change=probe_loss1 - probe_loss0))
                        is_primary = probe_index == 0 and direction_name == "actual" and alpha == 1
                        if is_primary:
                            primary_change = change

        loss1, acc1 = D.evaluate(model, data.test)
        common = dict(meta, checkpoint_id=checkpoint, checkpoint_step=state["step"],
                      test_loss_at_checkpoint=loss0, actual_signed_change=primary_change)
        horizon_rows.append(dict(common, target_step=state["step"], horizon_steps=0, test_loss_at_horizon=loss1,
                                 test_acc_at_horizon=acc1, test_loss_change=loss1 - loss0))

        horizons = {}
        for steps in a.horizon_steps:
            if steps > 0:
                horizons[steps] = np.nan
        for epochs in a.horizon_epochs:
            if epochs > 0:
                horizons[round(epochs * steps_per_epoch)] = epochs
        for steps, epochs in horizons.items():
            pending.append(dict(common, target_step=state["step"] + steps, horizon_epochs=epochs))

        save()
        print(f"[{rid}] epoch {epoch} checkpoint {checkpoint} test_acc {acc0:.4f}", flush=True)

    while pending and a.extend_for_horizons:
        for x, y in D.batches(data.train, cfg.batch_size, gen):
            advance(x, y)
            if not pending:
                break
    save()


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
    p.add_argument("--alphas", nargs="+", type=float, default=[-1, -.5, -.25, -.1, 0, .1, .25, .5, 1, 2])
    p.add_argument("--directions", nargs="+", default=["actual", "opposite", "random", "permuted", "other_batch", "random_labels"])
    p.add_argument("--random-directions", type=int, default=3)
    p.add_argument("--measure-checkpoints", nargs="*", type=int, default=[])
    p.add_argument("--measure-every", type=int, default=1)
    p.add_argument("--probe-sources", nargs="+", default=["val", "train"])
    p.add_argument("--probe-seeds", nargs="+", type=int, default=[12345, 12346, 12347, 12348, 12349])
    p.add_argument("--probe-sizes", nargs="+", type=int, default=[50, 100, 200, 500])
    p.add_argument("--class-balanced", choices=["no", "yes", "both"], default="both")
    p.add_argument("--horizon-steps", nargs="+", type=int, default=[1, 10, 100])
    p.add_argument("--horizon-epochs", nargs="+", type=float, default=[1, 2])
    p.add_argument("--extend-for-horizons", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    if 0 not in a.alphas or 1 not in a.alphas:
        raise SystemExit("--alphas must include 0 and 1")
    out = Path(a.out) if a.out else env.results_root() / "validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_config.json").write_text(json.dumps(vars(a), indent=2))
    tables = ([], [], [])
    for arch in a.archs:
        for depth in a.depths:
            for seed in a.seeds:
                width = a.mlp_width if arch == "mlp" else a.cnn_width
                val_size = 256 if a.fake else 5000
                cfg = RunConfig(dataset=a.dataset, fake=a.fake, val_size=val_size, arch=arch, depth=depth,
                                seed=seed, width=width, skip=a.skip, splits=(a.split,), lr=a.lr,
                                momentum=a.momentum, epochs=a.epochs, checkpoint_every=a.checkpoint_every, tag="val")
                data = get_data(cfg, a.device or env.device())
                run_one(cfg, a, data, out, tables)


if __name__ == "__main__":
    main()
