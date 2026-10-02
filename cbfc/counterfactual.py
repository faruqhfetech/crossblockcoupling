"""Four-point counterfactual: evaluate tr F_bb at (a_t,b_t), (a_t+1,b_t), (a_t,b_t+1),
(a_t+1,b_t+1). delta_a is the lower-induced change. For all-Linear b blocks,
delta_a = H + D + I with tr F_bb = mean_n sum_l U_ln V_ln, U = ||delta||^2, V = ||h||^2 + 1.
"""
import torch

from .fisher import block_stats, spectrum

POINTS = ("before", "a_only", "b_only", "joint")
MODEL_FISHER_POINTS = {"none": (), "delta_a": ("before", "a_only"), "all": POINTS}


def points(before, after, names_a, names_b):
    names_a, names_b = set(names_a), set(names_b)
    a_only, b_only = {}, {}
    for name, value in before.items():
        a_only[name] = after[name] if name in names_a else value
        b_only[name] = after[name] if name in names_b else value
    return {"before": before, "a_only": a_only, "b_only": b_only, "joint": after}


def decomposition(stats_before, stats_after, correct, N):
    dU = stats_after["U"] - stats_before["U"]
    dV = stats_after["V"] - stats_before["V"]
    H = stats_before["U"] * dV
    D = stats_before["V"] * dU
    I = dU * dV

    row = {"H_a": H.sum().item() / N, "D_a": D.sum().item() / N, "I_a": I.sum().item() / N}
    for layer in range(len(H)):
        row[f"H_a_L{layer}"] = H[layer].sum().item() / N
        row[f"D_a_L{layer}"] = D[layer].sum().item() / N
    for name, mask in (("right", correct), ("wrong", ~correct)):
        row[f"H_a_{name}"] = H[:, mask].sum().item() / N
        row[f"D_a_{name}"] = D[:, mask].sum().item() / N
    return row


def trace_deltas(stats, fisher_type):
    """Trace at each point and the three changes relative to 'before'."""
    key = f"{fisher_type}_trace_b"
    trace = {point: stats[point].get(key, float("nan")) for point in POINTS}
    delta_a = trace["a_only"] - trace["before"]
    delta_b = trace["b_only"] - trace["before"]
    delta_joint = trace["joint"] - trace["before"]

    row = {key: trace["before"]}
    for point in POINTS[1:]:
        row[f"{key}_{point}"] = trace[point]
    row[f"{fisher_type}_delta_a"] = delta_a
    row[f"{fisher_type}_delta_b"] = delta_b
    row[f"{fisher_type}_delta_joint"] = delta_joint
    row[f"{fisher_type}_residual"] = delta_joint - delta_a - delta_b
    return row


def four_point(model, buffers, before, after, split, x, y, model_fisher="delta_a", diagonal=False, loss="ce"):
    """model_fisher: none | delta_a (before and a_only) | all."""
    model_fisher_points = MODEL_FISHER_POINTS[model_fisher]
    params_at = points(before, after, split.names_a, split.names_b)
    stats = {}
    for point, params in params_at.items():
        stats[point] = block_stats(model, params, buffers, x, y, split.names_a, split.names_b,
                                   gram=(point == "before"), model_fisher=(point in model_fisher_points),
                                   diagonal=diagonal, loss=loss)

    fisher_types = ["emp"]
    if model_fisher_points:
        fisher_types.append("model")
    if diagonal:
        fisher_types.append("diag")
    row = {}
    for fisher_type in fisher_types:
        row.update(trace_deltas(stats, fisher_type))

    s0 = stats["before"]
    spec = spectrum(s0["emp_K_b"])
    row["emp_trace_a"] = s0["emp_trace_a"]
    row["emp_Fab_frob_sq"] = (s0["emp_K_a"] * s0["emp_K_b"]).sum().item()
    row["emp_top_eigenvalue_b"] = spec["top_eigenvalue"]
    row["emp_effective_rank_b"] = spec["effective_rank"]

    for layer, trace_before in enumerate(s0["emp_trace_b_layers"]):
        trace_a_only = stats["a_only"]["emp_trace_b_layers"][layer]
        row[f"emp_trace_b_L{layer}"] = trace_before
        row[f"emp_delta_a_L{layer}"] = trace_a_only - trace_before
        if diagonal:
            row[f"diag_trace_b_L{layer}"] = s0["diag_trace_b_layers"][layer]

    for point in POINTS:
        row[f"probe_loss_{point}"] = stats[point]["loss"].mean().item()
        row[f"probe_acc_{point}"] = stats[point]["correct"].float().mean().item()
    row["probe_dloss_a"] = row["probe_loss_a_only"] - row["probe_loss_before"]

    if "U" in s0:
        row.update(decomposition(s0, stats["a_only"], s0["correct"], len(x)))
    return row


def flat_change(before, after, names):
    changes = [(after[name] - before[name]).reshape(-1) for name in names]
    return torch.cat(changes)


def update_norms(before, after, split):
    step_a = flat_change(before, after, split.names_a)
    step_b = flat_change(before, after, split.names_b)
    weights_a = torch.cat([before[name].reshape(-1) for name in split.names_a])
    norms = dict(a_dot_norm=step_a.norm().item(),
                 b_dot_norm=step_b.norm().item(),
                 a_dot_norm_relative=(step_a.norm() / (weights_a.norm() + 1e-12)).item())
    return norms, step_a
