"""Causal LM on WikiText-103 (or --fake): four-point coupling per block split, per-sequence loss.

  python -m experiments.lm --seeds 0 1 2
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from cbfc import env
from cbfc.counterfactual import four_point, update_norms
from cbfc.io import write_rows
from cbfc.models import CausalLM
from cbfc.split import make_split


def sequences(tokens, length, limit=0):
    """Non-overlapping (input, next-token target) sequences of the given length."""
    n = (len(tokens) - 1) // length
    if limit:
        n = min(n, limit)
    t = torch.as_tensor(tokens[:n * length + 1].astype(np.int64))
    inputs = t[:-1].view(n, length)
    targets = t[1:].view(n, length)
    return inputs, targets


def token_loss(logits, targets, reduction="mean"):
    vocab = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), reduction=reduction)


@torch.no_grad()
def evaluate(model, x, y, bs=64):
    was_training = model.training
    model.eval()
    total = 0.0
    for start in range(0, len(x), bs):
        logits = model(x[start:start + bs])
        total += token_loss(logits, y[start:start + bs], reduction="sum").item()
    model.train(was_training)
    return total / y.numel()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--length", type=int, default=256)
    ap.add_argument("--train-sequences", type=int, default=0, help="0 = all")
    ap.add_argument("--valid-sequences", type=int, default=2000)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--splits", nargs="+", default=["depth_fraction", "param_count"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--checkpoint-every", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--probe-size", type=int, default=200)
    ap.add_argument("--probe-seed", type=int, default=12345)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    device = a.device or env.device()
    out = Path(a.out) if a.out else env.results_root() / "lm"
    out.mkdir(parents=True, exist_ok=True)

    if a.fake:
        rng = np.random.default_rng(0)
        train_tokens = rng.integers(3, a.vocab, 512 * a.length + 1)
        valid_tokens = rng.integers(3, a.vocab, 128 * a.length + 1)
    else:
        root = env.data_root() / "wikitext103"
        train_tokens = np.load(root / "wiki.train.npy")
        valid_tokens = np.load(root / "wiki.valid.npy")
        a.vocab = int(max(train_tokens.max(), valid_tokens.max())) + 1
    x_train, y_train = sequences(train_tokens, a.length, a.train_sequences)
    x_valid, y_valid = sequences(valid_tokens, a.length, a.valid_sequences)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_valid, y_valid = x_valid.to(device), y_valid.to(device)

    probe_rng = np.random.default_rng(a.probe_seed)
    probe_count = min(a.probe_size, len(x_valid))
    probe_idx = np.sort(probe_rng.choice(len(x_valid), probe_count, replace=False))
    probe_idx = torch.as_tensor(probe_idx, device=device)
    probe_x, probe_y = x_valid[probe_idx], y_valid[probe_idx]

    for seed in a.seeds:
        torch.manual_seed(seed)
        model = CausalLM(a.vocab, a.length, a.layers, a.dim, a.heads, dropout=a.dropout).to(device)
        splits = {mode: make_split(model, mode) for mode in a.splits}
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        dataset = "fake" if a.fake else "wikitext103"
        rid = f"lm_{dataset}_d{a.layers}_w{a.dim}_s{seed}"
        cfg = dict(vars(a), seed=seed, arch="lm", dataset=dataset, depth=a.layers, optimizer="adamw")
        (out / f"{rid}.json").write_text(json.dumps(cfg, indent=2))
        gen = torch.Generator().manual_seed(seed)
        rows = []
        t0 = time.time()
        step = 0

        def train_step(x, y):
            model.train()
            opt.zero_grad()
            loss = token_loss(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()
            return loss.item()

        for epoch in range(1, a.epochs + 1):
            perm = torch.randperm(len(x_train), generator=gen).to(device)
            loss_sum = 0
            seen = 0
            for start in range(0, len(perm), a.batch_size):
                idx = perm[start:start + a.batch_size]
                loss_sum += train_step(x_train[idx], y_train[idx]) * len(idx)
                seen += len(idx)
                step += 1
            if epoch % a.checkpoint_every:
                continue

            valid_loss = evaluate(model, x_valid, y_valid)
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            idx = torch.randint(len(x_train), (a.batch_size,), generator=gen).to(device)
            step_loss = train_step(x_train[idx], y_train[idx])
            after = {name: p.detach().clone() for name, p in model.named_parameters()}

            for mode, split in splits.items():
                row = dict(run_id=rid, condition=f"lm_d{a.layers}_w{a.dim}", arch="lm", dataset=dataset,
                           depth=a.layers, seed=seed, epoch=epoch, step=step, split=mode, split_cut=split.cut,
                           n_params_a=split.n_params_a, n_params_b=split.n_params_b,
                           train_loss_running=loss_sum / seen, test_loss=valid_loss, test_acc=math.nan,
                           valid_perplexity=math.exp(min(valid_loss, 20)), checkpoint_step_loss=step_loss)
                norms, _ = update_norms(before, after, split)
                row.update(norms)
                row.update(four_point(model, {}, before, after, split, probe_x, probe_y, "none", loss="lm"))
                rows.append(row)
            write_rows(out / f"{rid}.csv", rows)
            print(f"[{rid}] epoch {epoch} valid_loss {valid_loss:.4f} ({time.time() - t0:.0f}s)", flush=True)
        (out / f"{rid}.done").write_text(json.dumps(dict(seconds=time.time() - t0)))


if __name__ == "__main__":
    main()
