#!/usr/bin/env bash
# Experiment 1 (manuscript Appendix B): 8 scenario blocks simulated in parallel,
# then analysed into Tables 1-2 / Supplementary S2-S5 / Figures 1-4.
# All paths relative to repository root. Override via env: REPS, PY.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python3}"
REPS="${REPS:-2000}"
CFG=configs/run_config.yaml
SCN=configs/scenarios_exp1.yaml
OUT=outputs/replicates

mkdir -p "$OUT"
echo "[run_exp1] simulating 8 blocks x $REPS repetitions"
for b in 0 1 2 3 4 5 6 7; do
  "$PY" src/run_pipeline.py --mode simulate --config "$CFG" --scenarios "$SCN" \
    --block "$b" --repetitions "$REPS" --output-dir "$OUT" &
done
wait

echo "[run_exp1] analysing"
"$PY" src/run_pipeline.py --mode analyze --config "$CFG" --scenarios "$SCN" \
  --repetitions "$REPS" --output-dir "$OUT"
echo "[run_exp1] done"
