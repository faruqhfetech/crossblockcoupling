# Cross-Block Fisher Coupling

A network is split into a lower block *a* and an upper block *b*. At each checkpoint one real
optimizer step is taken, and the change in the upper block's empirical Fisher trace caused by
the lower-block part of that step is measured on a fixed probe set:

    delta_a = tr F_bb(a_{t+1}, b_t) - tr F_bb(a_t, b_t)

The project asks whether |delta_a| predicts learning in the next interval, and whether
learning-rate schedules, finite-history momentum and update allocation change it.

## Setup

```bash
pip install -r requirements.txt
```

Set `DATA_ROOT` in `.env` to a folder containing `MNIST/`, `cifar-10-batches-py/` and
`cifar-100-python/` (torchvision layout). Results are written to `RESULTS_ROOT`.

```bash
python tests/test_fisher.py
python run.py --preset smoke
```

## Experiments

```bash
python run.py --preset obs_cifar10
python run.py --dataset mnist --depths 5 9 --seeds 0 1 --epochs 10
python -m experiments.validation
python -m experiments.allocation
python -m experiments.lm_data
python -m experiments.lm
```

Presets: `obs_mnist`, `obs_cifar10`, `variants_cifar10`, `cnn_cifar10`, `sched_mnist`,
`sched_cifar10`, `momentum_history`, `momentum_gates`, `sim_mnist`, `sim_cifar10`,
`cnn_fisher_check`, `vit_cifar100`, `smoke`. Use `python run.py -h` for all options and
`--list` to print the planned runs. Completed runs are skipped.

## Analysis

```bash
python -m analysis.report     --runs results/obs_cifar10 --out results/report_obs
python -m analysis.structure  --runs results/obs_cifar10 --out results/report_structure
python -m analysis.momentum   --runs results/momentum_history --out results/report_momentum
python -m analysis.similarity --runs results/sim_cifar10 --out results/report_sim
python -m analysis.validation --input results/validation --out results/report_validation
python -m analysis.allocation --input results/allocation --out results/report_allocation
```

## Layout

```
cbfc/         models, data, block splits, Fisher statistics, four-point protocol,
              baselines, optimizers, schedules, configs, training loop
experiments/  validation, allocation, language model
analysis/     report, structure, momentum, similarity, validation, allocation
tests/        Fisher exactness tests
presets.py    experiment batteries
run.py        entry point
```
