"""Standalone evaluation entry point (manuscript Sec. 3.4).

Recomputes the full metric set from simulate.py replicate CSVs.

Usage:
  python3 scripts/evaluate.py --replicate-dir outputs/replicates \
      --config configs/run_config.yaml --scenarios configs/scenarios_exp1.yaml \
      --out outputs/metrics_eval.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import io, metrics  # noqa: E402
from src import simulate as sim  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replicate-dir", default="outputs/replicates")
    p.add_argument("--config", default="configs/run_config.yaml")
    p.add_argument("--scenarios", default="configs/scenarios_exp1.yaml")
    p.add_argument("--out", default="outputs/metrics_eval.csv")
    args = p.parse_args()

    cfg = io.load_yaml(args.config)
    delta = float(cfg["decision"]["delta"])
    root = io.resolve_path(args.replicate_dir)
    rows = []
    for path in sorted(root.glob("rep_*.csv")):
        recs = io.read_csv(str(path.relative_to(io.REPO_ROOT)))
        if not recs:
            continue
        scn = {"c_rate": float(recs[0].get("c_rate", 0.5)),
               "p10": float(recs[0].get("p10", 0.0)), "p00": float(recs[0].get("p00", 0.0)),
               "p11": float(recs[0].get("p11", 0.0)), "p01": float(recs[0].get("p01", 0.0))}
        truth = float(recs[0].get("truth", sim.analytic_truth(scn)))
        truth_t = float(recs[0].get("truth_treated", truth))
        for strategy in ["B0", "A1", "A2", "A3"]:
            ests = np.array([float(r[strategy]) for r in recs], dtype=float)
            ref = truth_t if strategy == "A2" else truth
            m = metrics.cell_metrics(ests, ref, delta)
            rows.append({"scenario": path.stem.replace("rep_", ""), "strategy": strategy, **m})
    out = io.write_csv(args.out, rows)
    print(f"[evaluate] wrote {len(rows)} rows -> {out.relative_to(io.REPO_ROOT)}")


if __name__ == "__main__":
    main()
