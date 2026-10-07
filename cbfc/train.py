"""Training loop. At each checkpoint: evaluate, take one retained step from an independent
stream, and run the four-point protocol per block split on the same network."""
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import data as D
from . import models, optim
from .counterfactual import four_point, points, update_norms
from .fisher import fisher_cosine, flatten, score_matrix
from .io import write_rows
from .schedules import Schedule
from .split import make_split

_CACHE = {}


def get_data(cfg, device):
    key = (cfg.dataset, cfg.fake, cfg.normalize, cfg.val_size, str(device))
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = D.load(cfg.dataset, device, cfg.val_size, cfg.normalize, cfg.fake)
    return _CACHE[key]


def build_model(cfg, data):
    if cfg.arch == "mlp":
        kwargs = dict(depth=cfg.depth, width=cfg.width, activation=cfg.activation, layernorm=cfg.layernorm,
                      dropout=cfg.dropout, res_scale=cfg.res_scale)
    elif cfg.arch == "rescnn":
        kwargs = dict(depth=cfg.depth, width=cfg.width, activation=cfg.activation, norm=cfg.norm,
                      groups=cfg.groups, dropout=cfg.dropout, skip=cfg.skip)
    elif cfg.arch == "vit":
        kwargs = dict(depth=cfg.depth, dim=cfg.width, heads=cfg.heads, patch=cfg.patch, mlp_ratio=cfg.mlp_ratio,
                      dropout=cfg.dropout)
    elif cfg.arch == "smallcnn":
        kwargs = dict(skip=cfg.skip)
    else:
        raise KeyError(cfg.arch)
    return models.build(cfg.arch, data.shape, data.classes, **kwargs)


def snapshot(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def train_step(model, opt, sched, x, y, names_b, cfg):
    model.train()
    opt.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    info = {}
    if isinstance(opt, optim.HistorySGD):
        x_gate = x[:cfg.gate_microbatch]
        y_gate = y[:cfg.gate_microbatch]
        gates, info = opt.decide(model, names_b, x_gate, y_gate)
        opt.step(gates)
    else:
        opt.step()
    sched.step()
    n_correct = (logits.argmax(1) == y).sum().item()
    return loss.item(), n_correct, info


def cos(u, v):
    return (u @ v / (u.norm() * v.norm())).item()


def history_compatibility(model, opt, params, buffers, x, y, names_b):
    """Probe compatibility of the applied update and each lag with the current gradient (block b)."""
    G, order = score_matrix(model, params, buffers, x, y, names_b)
    g_now = flatten(opt.last_current, order)
    update = flatten(opt.last_applied, order)
    out = dict(update_cos_current_b=cos(g_now, update),
               update_fisher_cos_current_b=fisher_cosine(G, g_now, update))
    n_lags = min(len(opt.last_lags[name]) for name in order)
    for k in range(1, n_lags + 1):
        lag = flatten({name: opt.last_lags[name][k - 1] for name in order}, order)
        out[f"lag{k}_cos_current_b"] = cos(g_now, lag)
        out[f"lag{k}_fisher_cos_current_b"] = fisher_cosine(G, g_now, lag)
    return out


def with_suffix(row, suffix):
    return {key + suffix: value for key, value in row.items()}


def save_checkpoint(path, epoch, before, after, buffers):
    state = {}
    for key, tensors in (("before", before), ("after", after), ("buffers", buffers)):
        state[key] = {name: value.cpu() for name, value in tensors.items()}
    state["epoch"] = epoch
    torch.save(state, path)


def run(cfg, out_dir, device="cuda", resume=True, log=print):
    out_dir = Path(out_dir)
    rid = cfg.run_id()
    out_dir.mkdir(parents=True, exist_ok=True)
    if resume and (out_dir / f"{rid}.done").exists():
        return log(f"[skip] {rid}")
    (out_dir / f"{rid}.json").write_text(cfg.to_json())
    (out_dir / f"{rid}_matrices.jsonl").unlink(missing_ok=True)

    torch.manual_seed(cfg.seed)
    data = get_data(cfg, device)
    model = build_model(cfg, data).to(device)

    splits = {}
    for mode in cfg.splits:
        cut = int(mode) if str(mode).isdigit() else mode
        splits[mode] = make_split(model, cut)
    main = next(iter(splits.values()))

    probe_x, probe_y, _ = D.probe(data, cfg.probe_source, cfg.probe_size, cfg.probe_seed)
    probes = {"": (probe_x, probe_y)}
    for source in cfg.extra_probes:
        extra_x, extra_y, _ = D.probe(data, source, cfg.probe_size, cfg.probe_seed)
        probes[f"_{source}"] = (extra_x, extra_y)

    opt = optim.build(model, cfg.optimizer, cfg.lr, cfg.momentum, cfg.weight_decay,
                      cfg.history_K, cfg.history_beta, cfg.gate, cfg.gate_threshold)
    steps_per_epoch = math.ceil(len(data.train) / cfg.batch_size)
    n_checkpoints = cfg.epochs // cfg.checkpoint_every
    total_steps = steps_per_epoch * cfg.epochs + n_checkpoints
    sched = Schedule(opt, cfg.schedule, cfg.lr, total_steps, cfg.lr_hi, cfg.lr_lo, cfg.lr_final)
    gen = torch.Generator().manual_seed(cfg.seed)
    stream = D.Stream(data.train, cfg.batch_size, cfg.seed + 1_000_000)
    history = isinstance(opt, optim.HistorySGD)
    rows = []
    prev_a_step = {}
    step = 0
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        loss_sum = 0
        correct_sum = 0
        seen = 0
        gates = []
        for x, y in D.batches(data.train, cfg.batch_size, gen):
            if cfg.augment:
                x = D.augment(x, gen)
            loss, n_correct, _ = train_step(model, opt, sched, x, y, main.names_b, cfg)
            loss_sum += loss * len(y)
            correct_sum += n_correct
            seen += len(y)
            step += 1
            if history and opt.K:
                gates.append(opt.last["gates"])
        if epoch % cfg.checkpoint_every:
            continue

        base = dict(run_id=rid, condition=cfg.condition_id(), seed=cfg.seed, epoch=epoch, step=step,
                    train_loss_running=loss_sum / seen, train_acc_running=correct_sum / seen)
        for split_name in ("val", "test"):
            loss, acc = D.evaluate(model, getattr(data, split_name))
            base[f"{split_name}_loss"] = loss
            base[f"{split_name}_acc"] = acc
        if gates:
            accept_rates = torch.tensor(gates, dtype=torch.float).mean(0).tolist()
            for k, rate in enumerate(accept_rates, 1):
                base[f"epoch_lag{k}_accept_rate"] = rate
        base["lr_a"], base["lr_b"] = sched.block_lrs(main.names_a)

        before = snapshot(model)
        buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
        x, y = stream.next()
        base["checkpoint_step_loss"], _, info = train_step(model, opt, sched, x, y, main.names_b, cfg)
        step += 1
        after = snapshot(model)

        if history:
            for key, value in {**opt.last, **info}.items():
                if key != "gates":
                    base[f"gate_{key}"] = value
            for k, kept in enumerate(opt.last["gates"], 1):
                base[f"lag{k}_kept"] = kept
            base.update(history_compatibility(model, opt, before, buffers, probe_x, probe_y, main.names_b))

        for mode, split in splits.items():
            norms, a_step = update_norms(before, after, split)
            if mode in prev_a_step:
                cos_prev = cos(a_step, prev_a_step[mode])
            else:
                cos_prev = float("nan")
            row = dict(base, split=str(mode), split_cut=split.cut, n_params_a=split.n_params_a,
                       n_params_b=split.n_params_b, cos_a_dot_prev=cos_prev, **norms)
            prev_a_step[mode] = a_step

            for suffix, (px, py) in probes.items():
                measured = four_point(model, buffers, before, after, split, px, py, cfg.model_fisher, cfg.diag_fisher)
                row.update(with_suffix(measured, suffix))
                if cfg.predictors:
                    from .predictors import baselines
                    params_at = points(before, after, split.names_a, split.names_b)
                    base_before = baselines(model, params_at["before"], buffers, px, py, split)
                    base_a_only = baselines(model, params_at["a_only"], buffers, px, py, split)
                    row.update(with_suffix(base_before, suffix))
                    for key in base_before:
                        row[f"d_{key}_a{suffix}"] = base_a_only[key] - base_before[key]
            rows.append(row)

        if cfg.save_matrices:
            from .predictors import similarity_matrices
            matrices = similarity_matrices(model, before, buffers, probe_x, main.names_a + main.names_b)
            with (out_dir / f"{rid}_matrices.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(dict(epoch=epoch, **matrices)) + "\n")
        if cfg.save_checkpoints:
            checkpoint_dir = out_dir / "checkpoints" / rid
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(checkpoint_dir / f"ep{epoch:03d}.pt", epoch, before, after, buffers)

        write_rows(out_dir / f"{rid}.csv", rows)
        main_row = rows[-len(splits)]
        elapsed = time.time() - t0
        log(f"[{rid}] ep {epoch}/{cfg.epochs} test_acc {base['test_acc']:.4f} "
            f"delta_a {main_row['emp_delta_a']:+.3e} ({elapsed:.0f}s)")

    summary = dict(seconds=time.time() - t0, rows=len(rows))
    (out_dir / f"{rid}.done").write_text(json.dumps(summary))
