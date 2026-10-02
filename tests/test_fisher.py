"""Exactness of fisher.py against brute-force per-example gradients (vmap(grad)).

Run: python -m pytest tests -q     (or: python tests/test_fisher.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from cbfc import models
from cbfc.counterfactual import four_point
from cbfc.fisher import block_stats, flatten, score_matrix
from cbfc.split import make_split

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
N, C = 12, 5


def brute_G(model, params, buffers, x, y, names):
    sel = {n: params[n] for n in names}
    rest = {n: v for n, v in params.items() if n not in sel}

    def lf(s, xi, yi):
        return F.cross_entropy(functional_call(model, ({**rest, **s}, buffers), (xi[None],)), yi[None])

    g = vmap(grad(lf), in_dims=(None, 0, 0))(sel, x, y)
    return torch.cat([g[n].reshape(len(x), -1) for n in names], 1)


CASES = {
    "mlp": lambda: (models.MLP(20, C, depth=4, width=8), (2, 10)),
    "mlp_ln_res_tanh": lambda: (models.MLP(20, C, depth=4, width=8, activation="tanh",
                                           layernorm=True, res_scale=0.3), (2, 10)),
    "mlp_linear": lambda: (models.MLP(20, C, depth=3, width=8, activation="linear"), (2, 10)),
    "rescnn_gn_skip": lambda: (models.ResCNN(3, C, depth=3, width=8, groups=4, skip=True), (3, 8, 8)),
    "rescnn_bn": lambda: (models.ResCNN(3, C, depth=2, width=8, norm="batchnorm"), (3, 8, 8)),
    "rescnn_nonorm": lambda: (models.ResCNN(3, C, depth=2, width=8, norm="none"), (3, 8, 8)),
    "smallcnn_skip": lambda: (models.SmallCNN(3, C, skip=True), (3, 8, 8)),
}


def setup(case):
    model, shape = CASES[case]()
    model = model.double()
    if case == "rescnn_bn":  # give BatchNorm non-trivial running stats
        model.train()
        with torch.no_grad():
            model(torch.randn(32, *shape))
    model.eval()
    x, y = torch.randn(N, *shape), torch.randint(C, (N,))
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    return model, params, buffers, x, y


def check(case):
    model, params, buffers, x, y = setup(case)
    split = make_split(model, "depth_fraction")
    s = block_stats(model, params, buffers, x, y, split.names_a, split.names_b)
    for tag, names in (("a", split.names_a), ("b", split.names_b)):
        G = brute_G(model, params, buffers, x, y, names)
        K = G @ G.T / N
        assert torch.allclose(s[f"emp_K_{tag}"], K, rtol=1e-8, atol=1e-10), (case, tag)
        assert abs(s[f"emp_trace_{tag}"] - K.trace().item()) < 1e-8 * max(1, K.trace().item())
    with torch.no_grad():
        p = functional_call(model, (params, buffers), (x,)).softmax(1)
    tr = sum((p[:, c] * brute_G(model, params, buffers, x, torch.full_like(y, c), split.names_b)
              .pow(2).sum(1)).sum() for c in range(C)) / N
    assert abs(s["model_trace_b"] - tr.item()) < 1e-8 * max(1, tr.item()), case
    G, order = score_matrix(model, params, buffers, x, y, split.names_b)
    assert torch.allclose(G, brute_G(model, params, buffers, x, y, order), rtol=1e-8, atol=1e-10)


def test_all_cases():
    for case in CASES:
        check(case)


def test_four_point_and_decomposition():
    model, params, buffers, x, y = setup("mlp")
    split = make_split(model, "depth_fraction")
    after = {n: v + 0.05 * torch.randn_like(v) for n, v in params.items()}
    snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
    row = four_point(model, buffers, params, after, split, x, y)
    for n, p in model.named_parameters():
        assert torch.equal(p, snapshot[n])
    assert abs(row["H_a"] + row["D_a"] + row["I_a"] - row["emp_delta_a"]) < 1e-9 * max(1, abs(row["emp_delta_a"]))
    tr = lambda prm: (brute_G(model, prm, buffers, x, y, split.names_b).pow(2).sum() / N).item()
    a_only = {n: after[n] if n in split.names_a else v for n, v in params.items()}
    assert abs(row["emp_delta_a"] - (tr(a_only) - tr(params))) < 1e-8
    _ = flatten(params, split.names_b)




def test_transformers():
    from cbfc.fisher import fisher_names
    for name, (model, x, y, loss) in {
        "vit": (models.ViT(3, C, depth=2, dim=8, heads=2, patch=4, image=8), torch.randn(N, 3, 8, 8),
                torch.randint(C, (N,)), "ce"),
        "lm": (models.CausalLM(11, 6, depth=2, dim=8, heads=2, dropout=0.0), torch.randint(11, (N, 6)),
               torch.randint(11, (N, 6)), "lm"),
    }.items():
        model = model.double().eval()
        params = {n: p.detach().clone() for n, p in model.named_parameters()}
        split = make_split(model, "depth_fraction")
        s = block_stats(model, params, {}, x, y, split.names_a, split.names_b,
                        model_fisher=(loss == "ce"), decomposition=False, loss=loss)
        nb = fisher_names(model, split.names_b)
        sel = {n: params[n] for n in nb}
        rest = {n: v for n, v in params.items() if n not in sel}

        def lf(sp, xi, yi):
            out = functional_call(model, {**rest, **sp}, (xi[None],))
            if loss == "ce":
                return F.cross_entropy(out, yi[None])
            return F.cross_entropy(out.reshape(-1, out.shape[-1]), yi.reshape(-1))

        g = vmap(grad(lf), in_dims=(None, 0, 0))(sel, x, y)
        G = torch.cat([g[n].reshape(N, -1) for n in nb], 1)
        assert torch.allclose(s["emp_K_b"], G @ G.T / N, rtol=1e-7, atol=1e-10), name


if __name__ == "__main__":
    test_all_cases()
    test_four_point_and_decomposition()
    test_transformers()
    print("all fisher tests passed")
