"""Recompute metrics from saved checkpoints; `python -m analysis.recompute DIR RUN_ID` self-checks."""
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from cbfc import data as D
from cbfc import env
from cbfc.config import RunConfig
from cbfc.counterfactual import four_point, points
from cbfc.split import make_split
from cbfc.train import build_model


@dataclass
class Run:
    cfg: RunConfig
    dir: Path
    model: torch.nn.Module
    split: object
    probe: tuple
    device: str

    def checkpoints(self):
        checkpoint_dir = self.dir / "checkpoints" / self.cfg.run_id()
        paths = sorted(checkpoint_dir.glob("ep*.pt"))
        if not paths:
            raise FileNotFoundError(f"no checkpoints found in {checkpoint_dir}")
        for path in paths:
            yield torch.load(path, map_location=self.device, weights_only=False)

    def points(self, ck):
        return points(ck["before"], ck["after"], self.split.names_a, self.split.names_b)


def load_run(run_dir, run_id, device=None, split=None) -> Run:
    device = device or env.device()
    run_dir = Path(run_dir)
    cfg = RunConfig.from_dict(json.loads((run_dir / f"{run_id}.json").read_text()))
    data = D.load(cfg.dataset, device, cfg.val_size, cfg.normalize, cfg.fake)
    model = build_model(cfg, data).to(device).eval()
    mode = split or cfg.splits[0]
    cut = int(mode) if str(mode).isdigit() else mode
    block_split = make_split(model, cut)
    probe_x, probe_y, _ = D.probe(data, cfg.probe_source, cfg.probe_size, cfg.probe_seed)
    return Run(cfg, run_dir, model, block_split, (probe_x, probe_y), device)


def self_check(run_dir, run_id):
    run = load_run(run_dir, run_id)
    csv = pd.read_csv(Path(run_dir) / f"{run_id}.csv", dtype={"split": str})
    csv = csv[csv.split == str(run.cfg.splits[0])].set_index("epoch")
    probe_x, probe_y = run.probe
    worst = 0.0
    for ck in run.checkpoints():
        row = four_point(run.model, ck["buffers"], ck["before"], ck["after"], run.split, probe_x, probe_y, "none")
        reference = csv.loc[ck["epoch"], "emp_delta_a"]
        relative_error = abs(row["emp_delta_a"] - reference) / max(abs(reference), 1e-12)
        worst = max(worst, relative_error)
    print(f"{run_id}: worst relative error vs CSV {worst:.2e}")
    return worst


if __name__ == "__main__":
    self_check(sys.argv[1], sys.argv[2])
