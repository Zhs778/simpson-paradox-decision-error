#!/usr/bin/env bash
# One-command reproduction: Experiment 1 + Experiment 3 + self-checks.
# NOTE: this runs the FULL simulation (48k + 96k replicates). For a sandbox
# smoke test set REPS=3 instead; that does NOT reproduce paper numbers.
set -euo pipefail
cd "$(dirname "$0")/.."

export PY="${PY:-python3}"
export REPS="${REPS:-2000}"

echo "=== run_all: verify (29 checks) ==="
"$PY" scripts/verify.py

echo "=== run_all: Experiment 1 ==="
bash scripts/run_exp1.sh

echo "=== run_all: evaluate + ablation + figures ==="
"$PY" scripts/evaluate.py --replicate-dir outputs/replicates --out outputs/metrics_eval.csv
"$PY" scripts/ablation.py --replicate-dir outputs/replicates --out outputs/ablation.csv
"$PY" scripts/plot.py --metrics outputs/metrics_eval.csv --out-dir outputs/figures

echo "=== run_all: Experiment 3 ==="
bash scripts/run_exp3.sh

echo "=== run_all: complete ==="
