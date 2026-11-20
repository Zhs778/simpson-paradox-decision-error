"""Experiment 3 runner: simulate 8 scenarios and write replicate-level CSVs.

Usage:
  python3 src/run_exp3.py --config configs/run_config.yaml \
      --scenarios configs/scenarios_exp3.yaml --output-dir outputs/exp3/replicates \
      --repetitions 2000 [--smoke]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import estimators as est  # noqa: E402
from src import simulate_exp3 as s3  # noqa: E402
from utils import io, seed  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/run_config.yaml")
    p.add_argument("--scenarios", default="configs/scenarios_exp3.yaml")
    p.add_argument("--output-dir", default="outputs/exp3/replicates")
    p.add_argument("--repetitions", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    cfg = io.load_yaml(args.config)
    sc = io.load_yaml(args.scenarios)
    reps = int(args.repetitions or cfg["simulation"]["repetitions"])
    if args.smoke:
        reps = min(reps, 3)
    seed_base = int(cfg["seeds"]["exp3_seed_base"])
    out = io.ensure_dir(args.output_dir)
    seed.set_global_seed(seed_base)

    for i, entry in enumerate(sc["scenario_order"]):
        level = sc["levels"][entry["level"]]
        n_cell = int(sc["stratum_sizes"][entry["size"]])
        seed_s = seed.scenario_seed(seed_base, i)
        truth = s3.analytic_truth_exp3(level)
        truth_t = s3.analytic_truth_treated_exp3(level)
        rows = []
        for r in range(reps):
            data = s3.sim_exp3_replicate(cfg, level, n_cell, seed.repetition_rng(seed_s, r))
            z1 = data["Z1"]
            vals = {
                "B0": est.crude_rd(data["A"], data["Y"]),
                "A1": est.standardize_sample(data["A"], z1, data["Y"]),
                "A2": est.standardize_treated(data["A"], z1, data["Y"]),
                "A3": est.gcomp_main_effects(data["A"], z1, data["Y"]),
                "A1Z1": est.standardize_z1_only(z1, data["A"], data["Y"]),
            }
            rows.append({"scenario": entry["id"], "level": entry["level"], "size": entry["size"],
                         "repetition": r, "truth": truth, "truth_treated": truth_t, **vals})
        rel = str(out.relative_to(io.REPO_ROOT) / f"rep_{entry['id']}.csv")
        io.write_csv(rel, rows)
        print(f"[exp3] {entry['id']}: {len(rows)} rows -> {rel}")


if __name__ == "__main__":
    main()
