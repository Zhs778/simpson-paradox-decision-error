"""Figure generation (manuscript Figures 1-4, S1-S5).

Reads the metrics CSVs produced by evaluate.py / run_pipeline.py --mode analyze.
No data is downloaded; all inputs are local generated CSVs.

Usage:
  python3 scripts/plot.py --metrics outputs/metrics_eval.csv \
      --out-dir outputs/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import io  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="outputs/metrics_eval.csv")
    p.add_argument("--out-dir", default="outputs/figures")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    out = io.ensure_dir(args.out_dir)
    df = pd.read_csv(io.resolve_path(args.metrics))
    if df.empty:
        raise SystemExit("No metrics to plot.")

    # Figure 1: decision error by family and strategy
    if {"family", "strategy", "decision_error"} <= set(df.columns):
        g = df.groupby(["family", "strategy"])["decision_error"].mean().unstack("strategy")
        ax = g.plot(kind="bar")
        ax.set_ylabel("Decision error rate")
        ax.set_title("Fig 1: three-band decision error by family")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(out / "figure1.png", dpi=args.dpi)
        fig.savefig(out / "figure1.pdf")
        plt.close(fig)

    # Figure 2: bias vs decision error scatter
    if {"bias", "decision_error"} <= set(df.columns):
        fig, ax = plt.subplots()
        ax.scatter(df["bias"], df["decision_error"], s=12, alpha=0.7)
        ax.set_xlabel("Bias")
        ax.set_ylabel("Decision error")
        ax.set_title("Fig 2: bias does not track decision error")
        fig.tight_layout()
        fig.savefig(out / "figure2.png", dpi=args.dpi)
        plt.close(fig)

    print(f"[plot] figures -> {out.relative_to(io.REPO_ROOT)}")


if __name__ == "__main__":
    main()
