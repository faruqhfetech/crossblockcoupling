"""Run configuration; ids hash every setting so runs never overwrite each other.

arch: mlp | rescnn | smallcnn | vit        activation: relu | tanh | gelu | linear
norm: groupnorm | batchnorm | none         optimizer: sgd | adamw | muon | history
gate: fixed | euclidean | fisher           schedule: none | cosine | linear | crossfade
probe_source: val | train | test           model_fisher: none | delta_a | all
"""
import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, fields

LISTS = ("splits", "extra_probes")
NOT_IN_CONDITION = ("seed", "depth", "save_checkpoints")


@dataclass
class RunConfig:
    dataset: str = "mnist"
    fake: bool = False
    normalize: bool = True
    val_size: int = 5000
    arch: str = "mlp"
    depth: int = 5
    width: int = 128
    activation: str = "relu"
    layernorm: bool = False
    dropout: float = 0.0
    res_scale: float = 0.0
    norm: str = "groupnorm"
    groups: int = 8
    skip: bool = False
    heads: int = 3
    patch: int = 4
    mlp_ratio: float = 4.0
    augment: bool = False
    splits: tuple = ("depth_fraction",)
    optimizer: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0
    history_K: int = 0
    history_beta: float = 0.0
    gate: str = "fixed"
    gate_threshold: float = 0.0
    gate_microbatch: int = 16
    schedule: str = "none"
    lr_hi: float | None = None
    lr_lo: float | None = None
    lr_final: float | None = None
    epochs: int = 30
    batch_size: int = 128
    checkpoint_every: int = 2
    probe_source: str = "val"
    probe_size: int = 200
    probe_seed: int = 12345
    model_fisher: str = "delta_a"
    extra_probes: tuple = ()
    diag_fisher: bool = False
    predictors: bool = False
    save_matrices: bool = False
    save_checkpoints: bool = False
    seed: int = 0
    tag: str = ""

    def condition_id(self):
        settings = {}
        for key, value in asdict(self).items():
            if key not in NOT_IN_CONDITION:
                settings[key] = value
        digest = hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]
        prefix = f"{self.tag}_" if self.tag else ""
        return f"{prefix}{self.arch}_{self.dataset}_{digest}"

    def run_id(self):
        return f"{self.condition_id()}_d{self.depth}_s{self.seed}"

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        kwargs = {}
        for key, value in d.items():
            if key not in known:
                continue
            if key in LISTS:
                value = tuple(value)
            kwargs[key] = value
        return cls(**kwargs)


def add_args(parser, multi=("depth", "seed")):
    for f in fields(RunConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.name in multi:
            parser.add_argument(flag + "s", type=int, nargs="+", default=[f.default], dest=f.name + "s")
        elif f.name in LISTS:
            parser.add_argument(flag, nargs="*", default=list(f.default))
        elif isinstance(f.default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            arg_type = float if f.default is None else type(f.default)
            parser.add_argument(flag, type=arg_type, default=f.default)
    return parser


def configs_from_args(args, multi=("depth", "seed")):
    """One RunConfig per combination of the multi-valued flags (depths x seeds)."""
    base = {}
    for f in fields(RunConfig):
        if f.name not in multi:
            base[f.name] = getattr(args, f.name)
    grid = [{}]
    for name in multi:
        new_grid = []
        for combo in grid:
            for value in getattr(args, name + "s"):
                new_grid.append({**combo, name: value})
        grid = new_grid
    return [RunConfig.from_dict({**base, **combo}) for combo in grid]
