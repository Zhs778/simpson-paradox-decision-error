"""Artifact self-checks (target: 29 checks as reported in Appendix B).

Runs only light structural/config/identity checks suitable for CI; does NOT
rerun full simulations. On a GPU machine use run_all.sh for the full pipeline.

Usage:
  python3 scripts/verify.py --config configs/run_config.yaml \
      --scenarios configs/scenarios_exp1.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import io  # noqa: E402
from src import simulate as sim  # noqa: E402
from src import simulate_exp3 as sim3  # noqa: E402
from utils import approximation as approx  # noqa: E402
from utils import metrics  # noqa: E402

CHECKS: list[tuple[str, callable]] = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check("required directories exist")
def c1(ctx):
    for d in ["src", "scripts", "configs", "utils"]:
        assert (io.REPO_ROOT / d).is_dir(), d


@check("required files exist")
def c2(ctx):
    for f in ["README.md", "requirements.txt", ".gitignore", "LICENSE"]:
        assert (io.REPO_ROOT / f).is_file(), f


@check("Exp1 has 24 scenarios")
def c3(ctx):
    assert len(ctx["sc"]["scenarios"]) == 24


@check("Exp1 family counts 13/5/6")
def c4(ctx):
    from collections import Counter
    counts = Counter(s["family"] for s in ctx["sc"]["scenarios"])
    assert counts == {"REV": 13, "CON": 5, "NULL": 6}, counts


@check("Exp1 eight blocks cover all scenarios")
def c5(ctx):
    ids = [i for b in ctx["sc"]["blocks"] for i in b]
    assert len(ids) == 24 and len(set(ids)) == 24


@check("SC18-20 closed-form truth")
def c6(ctx):
    for s in ctx["sc"]["scenarios"]:
        if s["id"] in ("SC18", "SC19", "SC20"):
            assert abs((0.02 - 0.22 * s["c_rate"]) - s["truth"]) < 1e-9


@check("all truths match analytic formula")
def c7(ctx):
    for s in ctx["sc"]["scenarios"]:
        assert abs(sim.analytic_truth(s) - s["truth"]) < 1e-3, s["id"]


@check("all treated truths finite")
def c8(ctx):
    for s in ctx["sc"]["scenarios"]:
        assert np.isfinite(sim.analytic_truth_treated(s)), s["id"]


@check("delta is 0.01")
def c9(ctx):
    assert abs(float(ctx["cfg"]["decision"]["delta"]) - 0.01) < 1e-12


@check("repetitions is 2000")
def c10(ctx):
    assert int(ctx["cfg"]["simulation"]["repetitions"]) == 2000


@check("seed scheme present")
def c11(ctx):
    for k in ["exp1_seed_base", "exp2_grid_base", "exp2_struct_base", "exp3_seed_base"]:
        assert k in ctx["cfg"]["seeds"], k


@check("frozen hash recorded")
def c12(ctx):
    assert ctx["cfg"].get("frozen_hash") == "c7ee2cd29ea87cc5"


@check("no absolute path in configs")
def c13(ctx):
    for f in ["run_config.yaml", "scenarios_exp1.yaml", "scenarios_exp3.yaml"]:
        text = (io.REPO_ROOT / "configs" / f).read_text(encoding="utf-8")
        assert "C:\\" not in text and "/Users/" not in text and "/home/" not in text, f


@check("Exp3 has 8 scenarios")
def c14(ctx):
    assert len(ctx["sc3"]["scenario_order"]) == 8


@check("Exp3 four levels")
def c15(ctx):
    assert len(ctx["sc3"]["levels"]) == 4


@check("Exp3 theta references match analytic")
def c16(ctx):
    for key, lv in ctx["sc3"]["levels"].items():
        assert abs(sim3.analytic_truth_exp3(lv) - lv["theta_reference"]) < 1e-9, key


@check("Exp3 near-threshold theta=0.025")
def c17(ctx):
    assert abs(sim3.analytic_truth_exp3(ctx["sc3"]["levels"]["L3"]) - 0.025) < 1e-9


@check("closed-band rule at +delta is uncertain")
def c18(ctx):
    assert metrics.recommend(0.01, 0.01) == 0


@check("closed-band rule at -delta is uncertain")
def c19(ctx):
    assert metrics.recommend(-0.01, 0.01) == 0


@check("below -delta is treat")
def c20(ctx):
    assert metrics.recommend(-0.02, 0.01) == -1


@check("above +delta is avoid")
def c21(ctx):
    assert metrics.recommend(0.02, 0.01) == 1


@check("sign error zero when signs agree")
def c22(ctx):
    assert metrics.sign_error([0.1, 0.2], 0.05) == 0.0


@check("decision error detects band crossing")
def c23(ctx):
    assert abs(metrics.decision_error([0.02, 0.005], 0.05, 0.01) - 0.5) < 1e-12


@check("Phi(0)=0.5")
def c24(ctx):
    assert abs(approx.phi(0.0) - 0.5) < 1e-12


@check("Phi error rises with estimator noise")
def c25(ctx):
    # larger sigma -> larger u-noise -> higher decision error
    e_small_sigma = approx.analytic_decision_error(0.05, 0.0, 0.02, 0.01)
    e_large_sigma = approx.analytic_decision_error(0.05, 0.0, 0.04, 0.01)
    assert e_large_sigma > e_small_sigma, (e_small_sigma, e_large_sigma)


@check("nomogram returns u and prediction")
def c26(ctx):
    n = approx.nomogram(0.03, 0.01, 0.02, 0.005)
    assert "u" in n and "predicted_decision_error" in n


@check("Exp2 grid has 36 cells")
def c27(ctx):
    from src import exp2
    assert len(exp2.grid_cells()) == 36


@check("Exp2 truth fixed at 0.05")
def c28(ctx):
    from src import exp2
    assert all(c["truth"] == 0.05 for c in exp2.grid_cells())


@check("RC rate MCSE formula")
def c29(ctx):
    assert abs(metrics.rate_mcse(0.5, 100) - 0.05) < 1e-12


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/run_config.yaml")
    p.add_argument("--scenarios", default="configs/scenarios_exp1.yaml")
    p.add_argument("--scenarios-exp3", default="configs/scenarios_exp3.yaml")
    args = p.parse_args()

    ctx = {"cfg": io.load_yaml(args.config), "sc": io.load_yaml(args.scenarios),
           "sc3": io.load_yaml(args.scenarios_exp3)}
    passed = 0
    for name, fn in CHECKS:
        try:
            fn(ctx)
            print(f"PASS  {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {name}: {exc}")
    print(f"\n{passed}/{len(CHECKS)} checks passed")
    sys.exit(0 if passed == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
