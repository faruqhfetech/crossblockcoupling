"""Exact empirical/model Fisher statistics of parameter blocks from one batched
forward-backward pass. Per-example gradients come in closed form from each
layer's input h and output gradient delta (Linear: delta h^T, Conv: unfolded,
Norm: delta * x_hat); block Grams K = G G^T / N share the spectrum of F = G^T G / N.
"""
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

SUPPORTED = (nn.Linear, nn.Conv2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d)
CHUNK = 16


@contextmanager
def eval_mode(model):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def fisher_names(model, names):
    excluded = set(getattr(model, "fisher_exclude", ()))
    return [name for name in names if name not in excluded]


def own_param_names(module_name, module):
    names = []
    for param_name, _ in module.named_parameters(recurse=False):
        if module_name:
            names.append(f"{module_name}.{param_name}")
        else:
            names.append(param_name)
    return names


def _owners(model, names):
    """The modules that own the given parameters, checked to be supported and whole."""
    wanted = set(names)
    owners = {}
    for module_name, module in model.named_modules():
        own = own_param_names(module_name, module)
        hit = [name for name in own if name in wanted]
        if not hit:
            continue
        if not isinstance(module, SUPPORTED):
            raise NotImplementedError(f"{module_name}: {type(module).__name__} not supported")
        if len(hit) != len(own):
            raise ValueError(f"{module_name}: parameters split across blocks")
        owners[module_name] = module

    covered = set()
    for module_name, module in owners.items():
        covered.update(own_param_names(module_name, module))
    if covered != wanted:
        raise ValueError(f"unhandled parameters: {wanted - covered}")
    return owners


def _xhat(module, h):
    """The normalized input that a norm layer's weight multiplies."""
    if isinstance(module, nn.GroupNorm):
        return F.group_norm(h, module.num_groups, eps=module.eps)
    if isinstance(module, nn.LayerNorm):
        return F.layer_norm(h, module.normalized_shape, eps=module.eps)
    shape = (1, -1) + (1,) * (h.dim() - 2)
    mean = module.running_mean.view(shape)
    std = torch.sqrt(module.running_var.view(shape) + module.eps)
    return (h - mean) / std


class Layer:
    """One layer's input h and output gradient delta, for all N probe examples."""

    def __init__(self, module, h, delta):
        self.mod = module
        self.h = h
        self.delta = delta
        self.bias = getattr(module, "bias", None) is not None

    def kron(self):
        return isinstance(self.mod, nn.Linear) and self.h.dim() == 2

    def _conv(self):
        """Unfolded input patches and flattened output gradients, in chunks of examples."""
        conv = self.mod
        if conv.groups != 1:
            raise NotImplementedError("grouped conv")
        for start in range(0, len(self.h), CHUNK):
            h_chunk = self.h[start:start + CHUNK]
            patches = F.unfold(h_chunk, conv.kernel_size, conv.dilation, conv.padding, conv.stride)
            delta_chunk = self.delta[start:start + CHUNK].flatten(2)
            yield patches, delta_chunk

    def _linear_dense(self):
        N = len(self.delta)
        out_features = self.delta.shape[-1]
        in_features = self.h.shape[-1]
        delta_rows = self.delta.reshape(N, -1, out_features)
        h_rows = self.h.reshape(N, -1, in_features)
        weight_grad = torch.bmm(delta_rows.transpose(1, 2), h_rows)
        parts = [weight_grad.reshape(N, -1)]
        if self.bias:
            bias_grad = delta_rows.sum(1)
            parts.append(bias_grad)
        return parts

    def _conv_dense(self):
        weight_grads, bias_grads = [], []
        for patches, delta_chunk in self._conv():
            weight_grad = torch.bmm(delta_chunk, patches.transpose(1, 2))
            weight_grads.append(weight_grad.flatten(1))
            bias_grads.append(delta_chunk.sum(2))
        parts = [torch.cat(weight_grads)]
        if self.bias:
            parts.append(torch.cat(bias_grads))
        return parts

    def _norm_dense(self):
        N = len(self.delta)
        delta = self.delta
        if isinstance(self.mod, nn.LayerNorm):
            n_normalized = len(self.mod.normalized_shape)
            summed_dims = tuple(range(1, delta.dim() - n_normalized))
        else:
            summed_dims = tuple(range(2, delta.dim()))
        weight_grad = delta * _xhat(self.mod, self.h)
        bias_grad = delta
        if summed_dims:
            weight_grad = weight_grad.sum(summed_dims)
            bias_grad = bias_grad.sum(summed_dims)
        return [weight_grad.reshape(N, -1), bias_grad.reshape(N, -1)]

    def dense(self):
        """(N, d) per-example gradient, weight then bias."""
        if isinstance(self.mod, nn.Linear):
            parts = self._linear_dense()
        elif isinstance(self.mod, nn.Conv2d):
            parts = self._conv_dense()
        else:
            parts = self._norm_dense()
        return torch.cat(parts, 1)

    def sqnorm(self):
        """Squared norm of each example's gradient, shape (N,)."""
        if self.kron():
            delta_sq = self.delta.pow(2).sum(1)
            h_sq = self.h.pow(2).sum(1)
            return delta_sq * (h_sq + self.bias)
        if isinstance(self.mod, nn.Conv2d):
            chunks = []
            for patches, delta_chunk in self._conv():
                weight_grad = torch.bmm(delta_chunk, patches.transpose(1, 2))
                weight_sq = weight_grad.pow(2).sum((1, 2))
                bias_sq = delta_chunk.sum(2).pow(2).sum(1)
                chunks.append(weight_sq + self.bias * bias_sq)
            return torch.cat(chunks)
        return self.dense().pow(2).sum(1)

    def sqnorm_diagonal(self):
        """O(P) conv approximation dropping cross-position terms; exact elsewhere."""
        if not isinstance(self.mod, nn.Conv2d):
            return self.sqnorm()
        chunks = []
        for patches, delta_chunk in self._conv():
            delta_sq = delta_chunk.pow(2).sum(1)
            patch_sq = patches.pow(2).sum(1)
            weight_sq = (delta_sq * patch_sq).sum(1)
            bias_sq = delta_sq.sum(1)
            chunks.append(weight_sq + self.bias * bias_sq)
        return torch.cat(chunks)

    def gram(self):
        """(N, N) Gram matrix of per-example gradients."""
        if self.kron():
            delta_gram = self.delta @ self.delta.T
            h_gram = self.h @ self.h.T
            return delta_gram * (h_gram + self.bias)
        G = self.dense()
        return G @ G.T


def capture(model, params, buffers, x, names):
    """Forward once; returns logits and {module: (module, input, output)}."""
    records = {}

    def make_hook(module_name):
        def hook(module, inputs, output):
            records[module_name] = (module, inputs[0].detach(), output)
        return hook

    hooks = []
    for module_name, module in _owners(model, names).items():
        hooks.append(module.register_forward_hook(make_hook(module_name)))
    try:
        live_params = {name: value.detach().requires_grad_() for name, value in params.items()}
        logits = functional_call(model, (live_params, buffers), (x,))
    finally:
        for hook in hooks:
            hook.remove()
    return logits, records


def per_example_loss(logits, labels, loss="ce"):
    if loss == "ce":
        return F.cross_entropy(logits, labels, reduction="none")
    N, T, V = logits.shape
    token_loss = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), reduction="none")
    return token_loss.view(N, T).mean(1)


def block_modules(records, names):
    """Recorded modules that own at least one of the given parameter names."""
    modules = []
    for module_name in records:
        prefix = module_name + "."
        if any(name.startswith(prefix) for name in names):
            modules.append(module_name)
    return modules


def block_stats(model, params, buffers, x, y, names_a, names_b,
                gram=True, model_fisher=True, decomposition=True, diagonal=False, loss="ce"):
    names_a = fisher_names(model, names_a)
    names_b = fisher_names(model, names_b)
    with eval_mode(model):
        N = len(x)
        logits, records = capture(model, params, buffers, x, names_a + names_b)
        modules = {"a": block_modules(records, names_a), "b": block_modules(records, names_b)}
        order = list(records)
        outputs = [records[module_name][2] for module_name in order]

        def layers_for(labels):
            total_loss = per_example_loss(logits, labels, loss).sum()
            deltas = torch.autograd.grad(total_loss, outputs, retain_graph=True)
            layers = {}
            for module_name, delta in zip(order, deltas):
                module, h, _ = records[module_name]
                layers[module_name] = Layer(module, h, delta.detach())
            return layers

        layers = layers_for(y)
        stats = {}
        for block in ("a", "b"):
            if gram:
                K = sum(layers[m].gram() for m in modules[block]) / N
                stats[f"emp_K_{block}"] = K
                stats[f"emp_trace_{block}"] = K.trace().item()
            else:
                total = sum(layers[m].sqnorm().sum() for m in modules[block])
                stats[f"emp_trace_{block}"] = total.item() / N

        stats["emp_trace_b_layers"] = [layers[m].sqnorm().sum().item() / N for m in modules["b"]]

        if diagonal:
            stats["diag_trace_b_layers"] = [layers[m].sqnorm_diagonal().sum().item() / N for m in modules["b"]]
            stats["diag_trace_b"] = sum(stats["diag_trace_b_layers"])

        all_linear = all(layers[m].kron() for m in modules["b"])
        if decomposition and all_linear:
            stats["U"] = torch.stack([layers[m].delta.pow(2).sum(1) for m in modules["b"]])
            stats["V"] = torch.stack([layers[m].h.pow(2).sum(1) + layers[m].bias for m in modules["b"]])

        if model_fisher and loss == "ce":
            probs = logits.detach().softmax(1)
            total = 0
            for c in range(logits.shape[1]):
                layers_c = layers_for(torch.full_like(y, c))
                sqnorm_c = sum(layers_c[m].sqnorm() for m in modules["b"])
                total += (probs[:, c] * sqnorm_c).sum().item()
            stats["model_trace_b"] = total / N

        logits = logits.detach()
        stats["loss"] = per_example_loss(logits, y, loss)
        hit = logits.argmax(-1) == y
        if loss == "ce":
            stats["correct"] = hit
        else:
            stats["correct"] = hit.float().mean(1) > 0.5
        return stats


def spectrum(K):
    eigenvalues = torch.linalg.eigvalsh(K.double()).clamp_min(0)
    sum_eig = eigenvalues.sum().item()
    sum_eig_sq = eigenvalues.pow(2).sum().item()
    effective_rank = sum_eig * sum_eig / sum_eig_sq if sum_eig_sq > 0 else 0.0
    return dict(top_eigenvalue=eigenvalues[-1].item(), effective_rank=effective_rank)


def fisher_cosine(G, u, v):
    """cos(u, v) under F = G^T G / N."""
    Gu = G @ u
    Gv = G @ v
    denominator = Gu.norm() * Gv.norm()
    if denominator <= 0:
        return float("nan")
    return (Gu @ Gv / denominator).item()


def score_matrix(model, params, buffers, x, y, names):
    """Dense per-example gradients (N, d) and the parameter names in column order."""
    names = fisher_names(model, names)
    with eval_mode(model):
        logits, records = capture(model, params, buffers, x, names)
        order = list(records)
        outputs = [records[module_name][2] for module_name in order]
        total_loss = F.cross_entropy(logits, y, reduction="sum")
        deltas = torch.autograd.grad(total_loss, outputs)
        columns, column_names = [], []
        for module_name, delta in zip(order, deltas):
            module, h, _ = records[module_name]
            columns.append(Layer(module, h, delta.detach()).dense())
            column_names += own_param_names(module_name, module)
        return torch.cat(columns, 1), column_names


def flatten(vec, order):
    return torch.cat([vec[name].reshape(-1) for name in order])
