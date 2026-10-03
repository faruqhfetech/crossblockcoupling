"""Per-step LR over depth-unit groups. crossfade: unit at depth d gets
(1-d) lower(t) + d upper(t); lower cosine-decays lr_hi -> lr_final, upper rises lr_lo -> lr_final.
Defaults: lr_hi = 10 lr, lr_lo = 0.1 lr, lr_final = lr."""
import math


class Schedule:
    def __init__(self, opt, kind="none", lr=0.01, total_steps=1, lr_hi=None, lr_lo=None, lr_final=None):
        if kind not in ("none", "cosine", "linear", "crossfade"):
            raise ValueError(kind)
        self.opt = opt
        self.kind = kind
        self.T = max(1, total_steps)
        self.lr = lr
        self.t = 0
        self.hi = lr * 10 if lr_hi is None else lr_hi
        self.lo = lr * 0.1 if lr_lo is None else lr_lo
        self.final = lr if lr_final is None else lr_final
        self.apply()

    def value(self, depth):
        fraction_done = min(self.t, self.T) / self.T
        cosine = 0.5 * (1 + math.cos(math.pi * fraction_done))
        lower_lr = self.final + (self.hi - self.final) * cosine
        if self.kind == "none":
            return self.lr
        if self.kind == "cosine":
            return lower_lr
        if self.kind == "linear":
            return self.hi + (self.final - self.hi) * fraction_done
        upper_lr = self.lo + (self.final - self.lo) * (1 - cosine)
        return (1 - depth) * lower_lr + depth * upper_lr

    def apply(self):
        for group in self.opt.param_groups:
            depth = group.get("depth_fraction", 0.5)
            group["lr"] = self.value(depth)

    def step(self):
        self.t += 1
        self.apply()

    def block_lrs(self, names_a):
        """Mean LR of the groups in block a and in block b."""
        names_a = set(names_a)
        lrs_a, lrs_b = [], []
        for group in self.opt.param_groups:
            in_a = bool(names_a & set(group.get("names", [])))
            if in_a:
                lrs_a.append(group["lr"])
            else:
                lrs_b.append(group["lr"])
        mean_a = sum(lrs_a) / len(lrs_a) if lrs_a else float("nan")
        mean_b = sum(lrs_b) / len(lrs_b) if lrs_b else float("nan")
        return mean_a, mean_b
