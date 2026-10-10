"""Cheap coupling baselines on the probe: forward/backward CKA, cosine and RMS, the
U-V norm product (tr F_bb minus a per-example covariance, Linear layers), and for MLPs
layer-Jacobian cosines and ||d logits / d h_(cut-1)||_F from JVPs."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

from .fisher import block_modules, capture, eval_mode, fisher_names
from .models import MLP


def linear_cka(X, Y):
    X = X.flatten(1)
    Y = Y.flatten(1)
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    cross = (X.T @ Y).pow(2).sum()
    norm_x = (X.T @ X).pow(2).sum().sqrt()
    norm_y = (Y.T @ Y).pow(2).sum().sqrt()
    denominator = (norm_x * norm_y).clamp_min(1e-20)
    return (cross / denominator).item()


def mean_cos(X, Y):
    if X.shape != Y.shape:
        return float("nan")
    return F.cosine_similarity(X.flatten(1), Y.flatten(1), dim=1).mean().item()


def rms(t):
    return t.pow(2).mean().sqrt().item()


def jacobian_stats(model, params, buffers, x, cut, n_dirs=8):
    with torch.no_grad():
        _, hidden = functional_call(model, (params, buffers), (x,), {"return_hidden": True})
    width = hidden[0].shape[1]
    directions = torch.randn(n_dirs, width, generator=torch.Generator().manual_seed(0)).to(x)

    def jvps(forward_kwargs, h):
        """J v for every direction v, stacked to (N, n_dirs, out)."""
        def submap(z):
            return functional_call(model, (params, buffers), (z,), forward_kwargs)
        products = []
        for v in directions:
            _, jv = jvp(submap, (h,), (v.expand_as(h),))
            products.append(jv)
        return torch.stack(products, 1)

    n_hidden = len(hidden)
    layer_jvps = {}
    for k in range(1, n_hidden):
        layer_jvps[k] = jvps({"layer": k}, hidden[k - 1])

    def jacobian_cos(i, j):
        return F.cosine_similarity(layer_jvps[i].flatten(1), layer_jvps[j].flatten(1), dim=1).mean().item()

    if n_hidden > 2:
        jac_adj = float(np.mean([jacobian_cos(k, k + 1) for k in range(1, n_hidden - 1)]))
    else:
        jac_adj = float("nan")
    if 2 <= cut < n_hidden:
        jac_boundary = jacobian_cos(cut - 1, cut)
    else:
        jac_boundary = float("nan")
    upper_jvps = jvps({"upper_from": cut}, hidden[cut - 1])
    frobenius = upper_jvps.pow(2).sum(2).mean(1).sqrt()
    return dict(jac_adj=jac_adj, jac_boundary=jac_boundary, jb_fro=frobenius.mean().item())


@torch.no_grad()
def similarity_matrices(model, params, buffers, x, names):
    with eval_mode(model):
        _, records = capture(model, params, buffers, x, fisher_names(model, names))
    outputs = [record[2].detach() for record in records.values()]
    cka = [[linear_cka(a, b) for b in outputs] for a in outputs]
    cos = [[mean_cos(a, b) for b in outputs] for a in outputs]
    return dict(modules=list(records), cka=cka, cos=cos, rms=[rms(o) for o in outputs])


def adjacent_mean(metric, tensors):
    values = [metric(tensors[i], tensors[i + 1]) for i in range(len(tensors) - 1)]
    return float(np.nanmean(values))


def baselines(model, params, buffers, x, y, split):
    with eval_mode(model):
        names = fisher_names(model, split.names_a + split.names_b)
        logits, records = capture(model, params, buffers, x, names)
        order = list(records)
        total_loss = F.cross_entropy(logits, y, reduction="sum")
        deltas = torch.autograd.grad(total_loss, [records[m][2] for m in order])
        deltas = [d.detach() for d in deltas]
        outputs = [records[m][2].detach() for m in order]
        inputs = [records[m][1] for m in order]
        b_modules = set(block_modules(records, split.names_b))
        b_index = [i for i, m in enumerate(order) if m in b_modules]
        first_b = b_index[0]

        stats = jacobian_stats(model, params, buffers, x, split.cut) if isinstance(model, MLP) else {}
        stats["cka_adj"] = adjacent_mean(linear_cka, outputs)
        stats["cos_adj"] = adjacent_mean(mean_cos, outputs)
        stats["cka_boundary"] = linear_cka(outputs[first_b - 1], outputs[first_b])
        stats["rms_b_input"] = rms(inputs[first_b])
        stats["bcka_adj"] = adjacent_mean(linear_cka, deltas)
        stats["brms_b_input"] = rms(deltas[first_b])

        rms_product = 0
        for i in b_index:
            delta_sq = deltas[i].flatten(1).pow(2).sum(1).mean().item()
            input_sq = (inputs[i].flatten(1).pow(2).sum(1) + 1).mean().item()
            rms_product += delta_sq * input_sq
        stats["rms_product"] = rms_product
        return stats
