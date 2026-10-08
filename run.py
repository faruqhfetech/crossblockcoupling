"""Run a preset battery or a custom grid; every RunConfig field is a flag.

  python run.py --preset obs_cifar10
  python run.py --dataset mnist --depths 5 9 --seeds 0 1 --epochs 10
"""
import argparse
import sys
from dataclasses import fields, replace
from pathlib import Path

from cbfc import env
from cbfc.config import RunConfig, add_args, configs_from_args
from cbfc.train import run
import presets


def flags_given_on_command_line(parser):
    """dest names of the options the user typed explicitly."""
    given = set()
    for action in parser._actions:
        if any(option in sys.argv for option in action.option_strings):
            given.add(action.dest)
    return given


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(presets.PRESETS))
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--list", action="store_true")
    add_args(parser)
    args = parser.parse_args()
    device = args.device or env.device()

    if args.preset:
        cfgs = presets.PRESETS[args.preset]()
        config_fields = {f.name for f in fields(RunConfig)}
        overrides = {}
        for name in flags_given_on_command_line(parser):
            if name in config_fields and name not in ("depth", "seed"):
                overrides[name] = getattr(args, name)
        cfgs = [replace(cfg, **overrides) for cfg in cfgs]
    else:
        cfgs = configs_from_args(args)

    if args.out:
        out = Path(args.out)
    else:
        out = env.results_root() / (args.preset or "runs")
    print(f"{len(cfgs)} runs -> {out} on {device}")
    for cfg in cfgs:
        if args.list:
            print(" ", cfg.run_id())
            continue
        run(cfg, out, device, resume=not args.no_resume)


if __name__ == "__main__":
    main()
