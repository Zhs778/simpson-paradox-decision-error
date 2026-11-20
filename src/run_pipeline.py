"""Experiment 1 simulation + analysis pipeline (manuscript Appendix B).

All options come from YAML config or command line; no absolute paths.

Usage (full run, from repository root):
  python3 src/run_pipeline.py --mode simulate \
      --scenarios configs/scenarios_exp1.yaml \
      --block 0 --output-dir outputs/replicates/block00 \
      --repetitions 2000
  python3 src/run_pipeline.py --mode analyze --output-dir outputs/replicates \
      --scenarios configs/scenarios_exp1.yaml --delta 0.01

Tiny smoke test (does NOT reproduce paper numbers):
  python3 src/run_pipeline.py --mode simulate --smoke --repetitions 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import estimators as est  # noqa: E402
from src import simulate as sim  # noqa: E402
from utils import io, metrics, seed  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproducibility pipeline (Experiment 1)")
    p.add_argument("--mode", choices=["simulate", "analyze", "check"], default="simulate")
    p.add_argument("--config", default="configs/run_config.yaml", help="YAML relative to repo root")
    p.add_argument("--scenarios", default="configs/scenarios_exp1.yaml", help="YAML relative to repo root")
    p.add_argument("--block", type=int, default=None, help="scenario block index (0-based)")
    p.add_argument("--output-dir", default="outputs/replicates", help="relative to repo root")
    p.add_argument("--repetitions", type=int, default=None, help="override YAML repetitions")
    p.add_argument("--delta", type=float, default=None, help="override decision threshold")
    p.add_argument("--seed-base", type=int, default=None, help="override exp1 seed base")
    p.add_argument("--smoke", action="store_true", help="3 scenarios x few reps; never writes large files")
    return p.parse_args(argv)


def resolve_settings(args: argparse.Namespace) -> dict:
    cfg = io.load_yaml(args.config)
    sc = io.load_yaml(args.scenarios)
    reps = args.repetitions if args.repetitions is not None else int(cfg["simulation"]["repetitions"])
    delta = args.delta if args.delta is not None else float(cfg["decision"]["delta"])
    seed_base = args.seed_base if args.seed_base is not None else int(cfg["seeds"]["exp1_seed_base"])
    return {"cfg": cfg, "sc": sc, "reps": reps, "delta": delta, "seed_base": seed_base}


def select_scenarios(sc: dict, block: int | None, smoke: bool) -> list[dict]:
    by_id = {s["id"]: s for s in sc["scenarios"]}
    if smoke:
        return [sc["scenarios"][i] for i in range(min(3, len(sc["scenarios"])))]
    if block is None:
        return sc["scenarios"]
    ids = sc["blocks"][block]
    return [by_id[i] for i in ids]


def simulate_mode(args: argparse.Namespace, settings: dict) -> None:
    scenarios = select_scenarios(settings["sc"], args.block, args.smoke)
    reps = min(settings["reps"], 3) if args.smoke else settings["reps"]
    out = io.ensure_dir(args.output_dir)
    seed.set_global_seed(settings["seed_base"])

    for idx, scn in enumerate(scenarios):
        seed_s = seed.scenario_seed(settings["seed_base"], idx)
        truth = sim.analytic_truth(scn)
        truth_t = sim.analytic_truth_treated(scn)
        rows = []
        for r in range(reps):
            rng = seed.repetition_rng(seed_s, r)
            data = sim.simulate_scenario(scn, rng)
            vals = est.run_all_strategies(data["A"], data["C"], data["Y"])
            vals["check_saturated_gcomp"] = est.saturated_gcomp_check(data["A"], data["C"], data["Y"])
            row = {"scenario": scn["id"], "family": scn["family"], "repetition": r,
                   "truth": truth, "truth_treated": truth_t}
            row.update(vals)
            rows.append(row)
        name = f"rep_{scn['id']}.csv"
        rel_out = str(out.relative_to(io.REPO_ROOT) / name)
        io.write_csv(rel_out, rows)
        print(f"[simulate] {scn['id']}: {len(rows)} rows -> {rel_out}")

    print(json.dumps({"mode": "simulate", "scenarios": len(scenarios), "repetitions_each": reps,
                      "delta": settings["delta"], "seed_base": settings["seed_base"]}, indent=2))


def analyze_mode(args: argparse.Namespace, settings: dict) -> None:
    rows = []
    for csv_path in sorted(Path(io.resolve_path(args.output_dir)).glob("rep_*.csv")):
        rows.extend(io.read_csv(str(csv_path.relative_to(io.REPO_ROOT))))
    if not rows:
        raise SystemExit("No replicate CSVs found; run --mode simulate first.")

    delta = settings["delta"]
    per_scn: dict[str, list[dict]] = {}
    for row in rows:
        per_scn.setdefault(row["scenario"], []).append(row)

    summary = []
    for scn_id, recs in sorted(per_scn.items()):
        truth = float(recs[0]["truth"])
        truth_t = float(recs[0]["truth_treated"])
        fam = recs[0]["family"]
        for strategy in ["B0", "A1", "A2", "A3"]:
            ests = np.array([float(r[strategy]) for r in recs], dtype=float)
            ref_truth = truth_t if strategy == "A2" else truth
            m = metrics.cell_metrics(ests, ref_truth, delta)
            d_s = metrics.paired_difference(
                ests, np.array([float(r["B0"]) for r in recs]), ref_truth, delta
            ) if strategy != "B0" else 0.0
            summary.append({"scenario": scn_id, "family": fam, "strategy": strategy,
                            "d_s_vs_B0": d_s, **m})
    out_path = io.write_csv("outputs/metrics_summary.csv", summary)
    print(f"[analyze] wrote {len(summary)} rows -> {out_path.relative_to(io.REPO_ROOT)}")

    # algebraic identity check: A1 == saturated g-comp
    diffs = [abs(float(r["A1"]) - float(r["check_saturated_gcomp"])) for r in rows
             if np.isfinite(float(r["A1"]))]
    if diffs:
        print(f"[check] max|A1 - saturated g-comp| = {max(diffs):.3e}")


def check_mode(args: argparse.Namespace, settings: dict) -> None:
    """Quick integrity checks on the config (no simulation)."""
    scns = settings["sc"]["scenarios"]
    fam_counts: dict[str, int] = {}
    issues = []
    for s in scns:
        fam_counts[s["family"]] = fam_counts.get(s["family"], 0) + 1
        analytic = sim.analytic_truth(s)
        if abs(analytic - float(s["truth"])) > 1e-3:
            issues.append(f"{s['id']}: truth {s['truth']} vs analytic {analytic:.4f}")
    print(json.dumps({"n_scenarios": len(scns), "families": fam_counts,
                      "truth_mismatches": issues}, indent=2))
    if issues:
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = resolve_settings(args)
    {"simulate": simulate_mode, "analyze": analyze_mode, "check": check_mode}[args.mode](args, settings)


if __name__ == "__main__":
    main()
