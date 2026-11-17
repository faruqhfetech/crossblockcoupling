"""Final-phase runs, one stage at a time. Each command's output goes to the screen and to <log>.log.

  python stages.py 1                      # baselines on the midterm networks
  python stages.py 4 --probe-size 500     # extra flags are added to every run.py command
  python stages.py 6 --with-lm            # also the language model
  python stages.py 3 --dry-run            # print the commands only
  python stages.py --list                 # every stage and its commands

Stages: 1 baselines, 2 CNNs on MNIST, 3 BatchNorm CNNs, 4 interventions, 5 side studies, 6 transformers.
A stage stops at the first failed command. Completed runs are skipped, so rerunning a stage resumes it.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Compiler caches go next to the code, not to the home directory: on the cluster the home is on
# AFS, which becomes read-only when the login token expires during long runs.
CACHE_DIRS = {"TRITON_CACHE_DIR": ROOT / ".cache" / "triton",
              "TORCHINDUCTOR_CACHE_DIR": ROOT / ".cache" / "torchinductor"}

STAGES = {
    1: ("baselines on the midterm networks", [
        ("base_obs_mnist", ["run.py", "--preset", "obs_mnist", "--predictors", "--save-checkpoints",
                            "--out", "results/base_obs_mnist"]),
        ("base_obs_cifar10", ["run.py", "--preset", "obs_cifar10", "--predictors", "--save-checkpoints",
                              "--out", "results/base_obs_cifar10"]),
        ("base_cnn_cifar10", ["run.py", "--preset", "cnn_cifar10", "--predictors", "--save-checkpoints",
                              "--out", "results/base_cnn_cifar10"]),
    ]),
    2: ("architecture vs learning stage: CNNs on MNIST", [
        ("cnn_mnist", ["run.py", "--preset", "cnn_cifar10", "--dataset", "mnist", "--out", "results/cnn_mnist"]),
    ]),
    3: ("normalization: BatchNorm CNNs", [
        ("cnn_bn", ["run.py", "--preset", "cnn_cifar10", "--norm", "batchnorm", "--out", "results/cnn_bn"]),
    ]),
    4: ("interventions", [
        ("sched_cifar10", ["run.py", "--preset", "sched_cifar10"]),
        ("obs_cifar10_mom", ["run.py", "--preset", "obs_cifar10", "--momentum", "0.9",
                             "--out", "results/obs_cifar10_mom"]),
        ("momentum_history", ["run.py", "--preset", "momentum_history"]),
        ("momentum_gates", ["run.py", "--preset", "momentum_gates"]),
    ]),
    5: ("side studies", [
        ("sim_cifar10", ["run.py", "--preset", "sim_cifar10"]),
        ("cnn_fisher_check", ["run.py", "--preset", "cnn_fisher_check"]),
    ]),
    6: ("transformers", [
        ("vit_test", ["run.py", "--preset", "vit_cifar100", "--seeds", "0", "--epochs", "2",
                      "--checkpoint-every", "1", "--out", "results/vit_test"]),
        ("vit", ["run.py", "--preset", "vit_cifar100"]),
    ]),
}

LM_COMMANDS = [
    ("lm_data", ["-m", "experiments.lm_data"]),
    ("lm", ["-m", "experiments.lm"]),
]


def commands_for(stage, with_lm):
    commands = list(STAGES[stage][1])
    if stage == 6 and with_lm:
        commands += LM_COMMANDS
    return commands


def full_command(args, extra):
    """python -u <args>, with the extra flags added to run.py commands only."""
    command = [sys.executable, "-u"] + args
    if args[0] == "run.py":
        command += extra
    return command


def run_logged(command, log_path):
    """Run a command, copy every output line to the screen and to the log; return its exit code."""
    environment = dict(os.environ)
    for name, path in CACHE_DIRS.items():
        if name not in environment:
            path.mkdir(parents=True, exist_ok=True)
            environment[name] = str(path)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", env=environment)
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return process.wait()


def print_stages():
    for stage, (title, _) in STAGES.items():
        print(f"stage {stage}: {title}")
        for log_name, args in commands_for(stage, with_lm=True):
            print(f"  {log_name:18s} python -u {' '.join(args)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", type=int, nargs="?", choices=sorted(STAGES))
    parser.add_argument("--with-lm", action="store_true", help="stage 6: also run the language model")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    args, extra = parser.parse_known_args()

    if args.list or args.stage is None:
        print_stages()
        return

    title = STAGES[args.stage][0]
    commands = commands_for(args.stage, args.with_lm)
    print(f"stage {args.stage}: {title} ({len(commands)} commands)", flush=True)
    for log_name, command_args in commands:
        command = full_command(command_args, extra)
        print("$ " + " ".join(command), flush=True)
        if args.dry_run:
            continue
        exit_code = run_logged(command, ROOT / f"{log_name}.log")
        if exit_code != 0:
            raise SystemExit(f"stopped: {log_name} failed with exit code {exit_code} (see {log_name}.log)")
    print(f"stage {args.stage} done", flush=True)


if __name__ == "__main__":
    main()
