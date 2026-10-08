"""Experiment batteries (lists of RunConfig). Depths follow the proposal."""
from cbfc.config import RunConfig

SPLITS = ("depth_fraction", "param_count")
MLP_DEPTHS = (5, 7, 9, 10, 12, 15)
CNN_DEPTHS = (3, 5, 7)
SEEDS3 = (0, 1, 2)


def grid(seeds=SEEDS3, depths=MLP_DEPTHS, **kw):
    cfgs = []
    for depth in depths:
        for seed in seeds:
            cfgs.append(RunConfig(depth=depth, seed=seed, **kw))
    return cfgs


def observational(dataset):
    return grid(dataset=dataset, splits=SPLITS, checkpoint_every=1, tag="obs")


def variants_cifar():
    variants = [dict(activation="tanh"), dict(activation="gelu"), dict(layernorm=True),
                dict(dropout=0.2), dict(weight_decay=1e-4),
                dict(optimizer="adamw", lr=3e-4), dict(optimizer="muon", lr=1e-3)]
    cfgs = []
    for variant in variants:
        cfgs += grid(dataset="cifar10", splits=SPLITS, tag="var", **variant)
    return cfgs


def cnn():
    cfgs = []
    for skip in (False, True):
        cfgs += grid(depths=CNN_DEPTHS, dataset="cifar10", arch="rescnn", width=64,
                     skip=skip, epochs=20, checkpoint_every=1, splits=SPLITS, tag="cnn")
    return cfgs


def schedules(dataset):
    cfgs = []
    for schedule in ("none", "cosine", "linear", "crossfade"):
        cfgs += grid(dataset=dataset, schedule=schedule, tag="sched")
    return cfgs


def momentum_history():
    conditions = [(0, 0.0)]
    for K in (1, 2, 3):
        for beta in (0.5, 0.7, 0.9):
            conditions.append((K, beta))
    cfgs = []
    for K, beta in conditions:
        cfgs += grid(seeds=range(5), depths=(7,), dataset="cifar10", arch="rescnn", skip=True,
                     optimizer="history", history_K=K, history_beta=beta,
                     epochs=20, checkpoint_every=1, tag="mom")
    return cfgs


def momentum_gates():
    methods = [dict(history_K=0), dict(history_K=3, history_beta=0.5),
               dict(history_K=3, history_beta=0.5, gate="euclidean"),
               dict(history_K=3, history_beta=0.5, gate="fisher")]
    cfgs = []
    for method in methods:
        cfgs += grid(seeds=range(5), depths=(7,), dataset="cifar10", arch="rescnn", skip=True,
                     optimizer="history", epochs=20, checkpoint_every=1, tag="gate", **method)
    return cfgs


def similarity(dataset):
    cfgs = []
    for res_scale in (0.0, 0.3, 1.0):
        cfgs += grid(seeds=range(8), depths=(6,), dataset=dataset, res_scale=res_scale,
                     epochs=30, checkpoint_every=1, save_checkpoints=True, predictors=True,
                     extra_probes=("train",), save_matrices=True, tag="sim")
    return cfgs


def cnn_fisher_check():
    cfgs = []
    for skip in (False, True):
        cfgs += grid(seeds=range(8), depths=(5,), dataset="cifar10", arch="smallcnn", skip=skip,
                     epochs=15, checkpoint_every=1, diag_fisher=True, predictors=True,
                     save_checkpoints=True, tag="cnncheck")
    return cfgs


def vit_cifar100():
    return grid(seeds=range(5), depths=(12,), dataset="cifar100", arch="vit", width=192, heads=3,
                optimizer="adamw", lr=3e-4, weight_decay=0.05, augment=True, epochs=100,
                checkpoint_every=5, model_fisher="none", splits=SPLITS, tag="vit")


def smoke():
    common = dict(fake=True, epochs=2, checkpoint_every=1, probe_size=16, val_size=64, tag="smoke")
    return [
        RunConfig(arch="mlp", depth=3, width=16, splits=SPLITS, **common),
        RunConfig(arch="mlp", depth=3, width=16, layernorm=True, schedule="crossfade", **common),
        RunConfig(arch="mlp", depth=3, width=16, optimizer="muon", lr=1e-3, **common),
        RunConfig(arch="rescnn", dataset="cifar10", depth=2, width=8, skip=True, **common),
        RunConfig(arch="rescnn", dataset="cifar10", depth=2, width=8, optimizer="history",
                  history_K=3, history_beta=0.5, gate="fisher", **common),
        RunConfig(arch="smallcnn", dataset="cifar10", save_checkpoints=True, predictors=True,
                  diag_fisher=True, **common),
        RunConfig(arch="vit", dataset="cifar100", depth=2, width=8, heads=2, optimizer="adamw",
                  lr=3e-4, augment=True, splits=SPLITS, **common),
        RunConfig(arch="mlp", depth=6, width=16, res_scale=0.3, predictors=True,
                  extra_probes=("train",), save_matrices=True, **common),
    ]


def obs_mnist():
    return observational("mnist")


def obs_cifar10():
    return observational("cifar10")


def sched_mnist():
    return schedules("mnist")


def sched_cifar10():
    return schedules("cifar10")


def sim_mnist():
    return similarity("mnist")


def sim_cifar10():
    return similarity("cifar10")


PRESETS = {
    "obs_mnist": obs_mnist,
    "obs_cifar10": obs_cifar10,
    "variants_cifar10": variants_cifar,
    "cnn_cifar10": cnn,
    "sched_mnist": sched_mnist,
    "sched_cifar10": sched_cifar10,
    "momentum_history": momentum_history,
    "momentum_gates": momentum_gates,
    "sim_mnist": sim_mnist,
    "sim_cifar10": sim_cifar10,
    "cnn_fisher_check": cnn_fisher_check,
    "vit_cifar100": vit_cifar100,
    "smoke": smoke,
}
