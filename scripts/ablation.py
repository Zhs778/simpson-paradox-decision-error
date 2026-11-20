"""Standalone ablation entry point.

Runs the three Exp1 contrasts (B0 vs A1, A1 vs A2, A1 vs A3) and the Exp2
boundary probes, reporting paired decision-error differences.

Usage:
  python3 scripts/ablation.py --replicate-dir outputs/replicates \
      --config configs/run_config.yaml --out outputs/ablation.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import io, metrics  # noqa: E402


CONTRASTS = [("B0", "A1"), ("A1", "A2"), ("A1", "A3")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replicate-dir", default="outputs/replicates")
    p.add_argument("--config", default="configs/run_config.yaml")
    p.add_argument("--out", default="outputs/ablation.csv")
    args = p.parse_args()

    cfg = io.load_yaml(args.config)
    delta = float(cfg["decision"]["delta"])
    root = io.resolve_path(args.replicate_dir)
    rows = []
    for path in sorted(root.glob("rep_*.csv")):
        recs = io.read_csv(str(path.relative_to(io.REPO_ROOT)))
        for base, alt in CONTRASTS:
            truth = float(recs[0]["truth"])
            d = metrics.paired_difference(
                np.array([float(r[alt]) for r in recs]),
                np.array([float(r[base]) for r in recs]), truth, delta)
            rows.append({"scenario": path.stem.replace("rep_", ""),
                         "contrast": f"{alt}-vs-{base}", "paired_decision_diff": d})
    out = io.write_csv(args.out, rows)
    print(f"[ablation] wrote {len(rows)} rows -> {out.relative_to(io.REPO_ROOT)}")


if __name__ == "__main__":
    main()
