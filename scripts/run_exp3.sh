#!/usr/bin/env bash
# Experiment 3 (Sec. 3.7): 8 scenarios x 2000 reps -> Table 6 / Supp S8.
# Override via env: REPS, PY.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python3}"
REPS="${REPS:-2000}"
CFG=configs/run_config.yaml
SCN=configs/scenarios_exp3.yaml
OUT=outputs/exp3

mkdir -p "$OUT"
echo "[run_exp3] simulating 8 scenarios x $REPS repetitions"
"$PY" src/run_exp3.py --config "$CFG" --scenarios "$SCN" \
  --output-dir "$OUT/replicates" --repetitions "$REPS"
echo "[run_exp3] analysing"
"$PY" scripts/evaluate.py --replicate-dir "$OUT/replicates" --config "$CFG" \
  --out "$OUT/metrics_exp3.csv"
echo "[run_exp3] done"
