"""Block split over depth units: depth_fraction (half the units), param_count, or an int cut."""
from dataclasses import dataclass


@dataclass
class BlockSplit:
    cut: int
    names_a: list
    names_b: list
    n_params_a: int
    n_params_b: int


def param_count_cut(unit_sizes):
    """The cut that makes the lower block's parameter count closest to half the total."""
    total = sum(unit_sizes)
    best_cut, best_gap = None, None
    for cut in range(1, len(unit_sizes)):
        lower = sum(unit_sizes[:cut])
        gap = abs(2 * lower - total)
        if best_gap is None or gap < best_gap:
            best_cut, best_gap = cut, gap
    return best_cut


def make_split(model, mode="depth_fraction"):
    units = model.units()
    params = dict(model.named_parameters())
    unit_sizes = [sum(params[name].numel() for name in unit) for unit in units]

    if isinstance(mode, int):
        cut = mode
    elif mode == "depth_fraction":
        cut = max(1, len(units) // 2)
    elif mode == "param_count":
        cut = param_count_cut(unit_sizes)
    else:
        raise ValueError(mode)
    if not 1 <= cut < len(units):
        raise ValueError(f"cut {cut} invalid for {len(units)} units")

    names_a = [name for unit in units[:cut] for name in unit]
    names_b = [name for unit in units[cut:] for name in unit]
    missing = set(params) - set(names_a) - set(names_b)
    if missing:
        raise RuntimeError(f"units miss {missing}")
    return BlockSplit(cut, names_a, names_b, sum(unit_sizes[:cut]), sum(unit_sizes[cut:]))
