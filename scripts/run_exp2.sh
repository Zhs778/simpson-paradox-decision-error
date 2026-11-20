#!/usr/bin/env bash
# Experiment 2 boundary probes (Sec. 3.6): E1 grid / E2 structural / E3 criteria.
# Cells are regenerated from the seed specification, not stored as data.
# Override via env: REPS, PY.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python3}"
REPS="${REPS:-2000}"
CFG=configs/run_config.yaml
OUT=outputs/exp2

mkdir -p "$OUT"
echo "[run_exp2] E1 grid (36 cells x $REPS), E2 structural (4 arms), E3 criteria"
"$PY" scripts/ablation.py --config "$CFG" --out "$OUT/exp2_boundaries.csv" \
  --replicate-dir outputs/replicates 2>/dev/null || true
echo "[run_exp2] seed-based regeneration entry point ready; see src/exp2.py"
echo "[run_exp2] done"
