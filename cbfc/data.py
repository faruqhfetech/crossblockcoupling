"""In-memory datasets with a seeded validation split (held out from train) and fixed probes."""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call
from torchvision import datasets

from . import env
from .fisher import eval_mode

INFO = {
    "mnist": dict(shape=(1, 28, 28), classes=10, mean=(0.1307,), std=(0.3081,), cls=datasets.MNIST),
    "cifar10": dict(shape=(3, 32, 32), classes=10, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616),
                    cls=datasets.CIFAR10),
    "cifar100": dict(shape=(3, 32, 32), classes=100, mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762),
                     cls=datasets.CIFAR100),
}


@dataclass
class Split:
    x: torch.Tensor
    y: torch.Tensor

    def __len__(self):
        return len(self.y)


@dataclass
class Data:
    name: str
    classes: int
    shape: tuple
    train: Split
    val: Split
    test: Split


def to_nchw(x):
    """Grayscale (N, H, W) gets a channel axis; color (N, H, W, C) moves channels first."""
    if x.dim() == 3:
        return x.unsqueeze(1)
    return x.permute(0, 3, 1, 2)


def _raw(name, train):
    npz_path = env.data_root() / f"{name}.npz"
    if npz_path.exists():
        arrays = np.load(npz_path)
        x_key = "x_train" if train else "x_test"
        y_key = "y_train" if train else "y_test"
        x = torch.from_numpy(arrays[x_key]).float().div(255)
        y = torch.from_numpy(arrays[y_key]).long()
        return to_nchw(x), y
    dataset = INFO[name]["cls"](str(env.data_root()), train=train, download=False)
    x = torch.as_tensor(dataset.data).float().div(255)
    y = torch.as_tensor(dataset.targets).long()
    return to_nchw(x), y


def _fake(name, n, seed):
    generator = torch.Generator().manual_seed(seed)
    shape = INFO[name]["shape"]
    classes = INFO[name]["classes"]
    x = torch.rand((n, *shape), generator=generator)
    y = torch.randint(classes, (n,), generator=generator)
    return x, y


def load(name, device, val_size=5000, normalize=True, fake=False, split_seed=0):
    info = INFO[name]
    if fake:
        x_train, y_train = _fake(name, 2048, 1)
        x_test, y_test = _fake(name, 512, 2)
    else:
        x_train, y_train = _raw(name, True)
        x_test, y_test = _raw(name, False)

    if normalize:
        mean = torch.tensor(info["mean"]).view(1, -1, 1, 1)
        std = torch.tensor(info["std"]).view(1, -1, 1, 1)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std

    perm = torch.randperm(len(y_train), generator=torch.Generator().manual_seed(split_seed))
    n_val = min(val_size, len(y_train) // 4)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    def on_device(x, y):
        return Split(x.to(device), y.to(device))

    train = on_device(x_train[train_idx], y_train[train_idx])
    val = on_device(x_train[val_idx], y_train[val_idx])
    test = on_device(x_test, y_test)
    return Data(name, info["classes"], info["shape"], train, val, test)


def probe(data, source="val", size=200, seed=12345, balanced=False):
    split = getattr(data, source)
    rng = np.random.default_rng(seed)
    if balanced:
        labels = split.y.cpu().numpy()
        per_class, extra = divmod(size, data.classes)
        chosen = []
        for c in range(data.classes):
            candidates = np.flatnonzero(labels == c)
            n_take = per_class + (c < extra)
            chosen.append(rng.choice(candidates, n_take, replace=False))
        idx = np.concatenate(chosen)
    else:
        idx = rng.choice(len(split), size=size, replace=False)
    idx_tensor = torch.as_tensor(np.sort(idx), device=split.y.device)
    return split.x[idx_tensor], split.y[idx_tensor], idx_tensor.cpu().numpy()


def batches(split, batch_size, generator):
    perm = torch.randperm(len(split), generator=generator).to(split.y.device)
    for start in range(0, len(perm), batch_size):
        idx = perm[start:start + batch_size]
        yield split.x[idx], split.y[idx]


def augment(x, generator, pad=4):
    """Random crop (zero pad) and horizontal flip."""
    N, _, H, W = x.shape
    x_padded = F.pad(x, (pad,) * 4)
    offsets = torch.randint(0, 2 * pad + 1, (2, N), generator=generator).tolist()
    row_offsets, col_offsets = offsets[0], offsets[1]
    crops = []
    for n in range(N):
        top = row_offsets[n]
        left = col_offsets[n]
        crops.append(x_padded[n, :, top:top + H, left:left + W])
    out = torch.stack(crops)
    flip = (torch.rand(N, generator=generator) < 0.5).to(x.device)
    out[flip] = out[flip].flip(-1)
    return out


class Stream:
    """Endless shuffled batches, independent of the epoch order."""

    def __init__(self, split, batch_size, seed):
        self.split = split
        self.bs = batch_size
        self.gen = torch.Generator().manual_seed(seed)
        self.it = iter(())

    def next(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = batches(self.split, self.bs, self.gen)
            return next(self.it)


@torch.no_grad()
def evaluate(model, split, params=None, buffers=None, batch_size=2000):
    total_loss = 0.0
    correct = 0.0
    with eval_mode(model):
        for start in range(0, len(split), batch_size):
            x = split.x[start:start + batch_size]
            y = split.y[start:start + batch_size]
            if params is None:
                logits = model(x)
            else:
                logits = functional_call(model, (params, buffers or {}), (x,))
            total_loss += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(1) == y).sum().item()
    return total_loss / len(split), correct / len(split)
