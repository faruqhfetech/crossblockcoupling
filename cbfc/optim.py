"""sgd, adamw, muon (MLP hidden weights; AdamW for the rest), and finite-history SGD:
u_t = g_t + sum_k beta^k g_{t-k}, where lag k is kept iff gate is fixed or the
(Euclidean | Fisher) cosine of its block-b part with g_t,b exceeds threshold.
"""
import copy
from collections import deque

import torch
import torch.nn.functional as F

from .fisher import fisher_cosine, flatten, score_matrix


def param_groups(model):
    """One optimizer group per depth unit, tagged with its names and relative depth."""
    params = dict(model.named_parameters())
    units = model.units()
    last_index = max(len(units) - 1, 1)
    groups = []
    for i, unit in enumerate(units):
        groups.append({"params": [params[name] for name in unit],
                       "names": list(unit),
                       "depth_fraction": i / last_index})
    return groups


class MultiOptimizer:
    def __init__(self, opts):
        self.opts = opts
        self.param_groups = [group for opt in opts for group in opt.param_groups]

    def zero_grad(self, set_to_none=True):
        for opt in self.opts:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        for opt in self.opts:
            opt.step()


class HistorySGD:
    def __init__(self, groups, lr, K=0, beta=0.0, weight_decay=0.0, gate="fixed", threshold=0.0):
        self.param_groups = [dict(group, lr=lr) for group in groups]
        self.params = {}
        for group in self.param_groups:
            for name, param in zip(group["names"], group["params"]):
                self.params[name] = param
        self.K = K
        self.beta = beta
        self.wd = weight_decay
        self.gate = gate
        self.threshold = threshold
        self.hist = {name: deque(maxlen=K) for name in self.params}
        self.last = {}

    def zero_grad(self, set_to_none=True):
        for param in self.params.values():
            param.grad = None

    def grads(self):
        """Current gradient plus weight decay, for every parameter that has a gradient."""
        grads = {}
        for name, param in self.params.items():
            if param.grad is None:
                continue
            grads[name] = param.grad.detach() + self.wd * param.detach()
        return grads

    def decide(self, model, names_b, x=None, y=None):
        """Which lags to keep (1 = keep), and the cosines that decided it."""
        present = []
        for k in range(1, self.K + 1):
            all_have_lag = all(len(history) >= k for history in self.hist.values())
            present.append(all_have_lag)
        if self.gate == "fixed" or not self.K:
            return [int(p) for p in present], {}

        current = self.grads()
        if self.gate == "fisher":
            params = {name: p.detach() for name, p in model.named_parameters()}
            buffers = dict(model.named_buffers())
            G, order = score_matrix(model, params, buffers, x, y, names_b)
        else:
            G, order = None, list(names_b)
        g_now = flatten(current, order)

        info, gates = {}, []
        for k in range(1, self.K + 1):
            if not present[k - 1]:
                gates.append(0)
                continue
            g_lag = flatten({name: self.hist[name][k - 1] for name in order}, order)
            euclidean_cos = (g_now @ g_lag / (g_now.norm() * g_lag.norm())).item()
            fisher_cos = fisher_cosine(G, g_now, g_lag) if G is not None else float("nan")
            info[f"lag{k}_euclidean_cos"] = euclidean_cos
            info[f"lag{k}_fisher_cos"] = fisher_cos
            score = fisher_cos if self.gate == "fisher" else euclidean_cos
            gates.append(int(score > self.threshold))
        return gates, info

    @torch.no_grad()
    def step(self, gates=None):
        if gates is None:
            gates = [1] * self.K
        lr_of = {}
        for group in self.param_groups:
            for name in group["names"]:
                lr_of[name] = group["lr"]

        self.last_current, self.last_applied, self.last_lags = {}, {}, {}
        candidate_sq = 0.0
        applied_sq = 0.0
        for name, g in self.grads().items():
            lags = []
            for k, past in enumerate(self.hist[name], 1):
                lags.append(self.beta ** k * past)

            update = torch.zeros_like(g)
            for lag, keep in zip(lags, gates):
                if keep:
                    update = update + lag
            update = g + update

            all_lags = torch.zeros_like(g)
            for lag in lags:
                all_lags = all_lags + lag
            candidate = g + all_lags

            candidate_sq += float(candidate.double().pow(2).sum())
            applied_sq += float(update.double().pow(2).sum())
            self.params[name].add_(update, alpha=-lr_of[name])
            self.last_current[name] = g
            self.last_applied[name] = update
            self.last_lags[name] = lags
            self.hist[name].appendleft(g)

        kept_weight = 0
        for k, keep in enumerate(gates, 1):
            kept_weight += self.beta ** k * keep
        self.last = dict(gates=list(gates), candidate_norm=candidate_sq ** 0.5, applied_norm=applied_sq ** 0.5,
                         kept_lags=sum(gates), kept_weight=kept_weight)


def fork(model, opt):
    """Independent copy of a model and its optimizer (state included)."""
    return copy.deepcopy((model, opt))


def step_delta(model, opt, x, y):
    """One real train-mode step; returns the parameter change."""
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    model.train()
    opt.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    return {name: p.detach() - before[name] for name, p in model.named_parameters()}


def build(model, name="sgd", lr=0.01, momentum=0.0, weight_decay=0.0, K=0, beta=0.0, gate="fixed", threshold=0.0):
    groups = param_groups(model)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)
    if name == "history":
        return HistorySGD(groups, lr, K, beta, weight_decay, gate, threshold)
    if name == "muon":
        params = dict(model.named_parameters())
        if any(p.dim() > 2 for p in params.values()):
            raise NotImplementedError("muon is implemented for MLPs only")
        hidden_weights = {unit[0] for unit in model.units()[:-1]}
        muon_params = [p for n, p in params.items() if n in hidden_weights]
        adamw_params = [p for n, p in params.items() if n not in hidden_weights]
        muon = torch.optim.Muon(muon_params, lr=lr, weight_decay=weight_decay)
        adamw = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay)
        return MultiOptimizer([muon, adamw])
    raise ValueError(name)
