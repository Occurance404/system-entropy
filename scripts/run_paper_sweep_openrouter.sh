#!/usr/bin/env bash
set -euo pipefail

# Paper sweep for OpenRouter (API-backed).
#
# Usage:
#   ./scripts/run_paper_sweep_openrouter.sh [models_json] [suite_json] [repeats]
#
# Prereq:
#   export OPENROUTER_API_KEY=...
#
# Recommended env:
#   export SANDBOX_BACKEND=local
#   export SANDBOX_PER_RUN=1
#   export SCENARIO_SEED=0
#   export REQUEST_TOOLS=auto
#   export REQUEST_LOGPROBS=auto

MODELS_JSON="${1:-benchmarks/models.openrouter.paper.template.json}"
SUITE_JSON="${2:-benchmarks/suite_paper_v1.json}"
REPEATS="${3:-10}"

if [[ -x "./.venv/bin/python" ]]; then
  PY="./.venv/bin/python"
else
  PY="python3"
fi

export SANDBOX_BACKEND="${SANDBOX_BACKEND:-local}"
export SANDBOX_PER_RUN="${SANDBOX_PER_RUN:-1}"
export SCENARIO_SEED="${SCENARIO_SEED:-0}"
export REQUEST_TOOLS="${REQUEST_TOOLS:-auto}"
export REQUEST_LOGPROBS="${REQUEST_LOGPROBS:-auto}"
export SCR_LOCAL_FILES_ONLY="${SCR_LOCAL_FILES_ONLY:-1}"

echo "OpenRouter paper sweep"
echo "  python:   $PY"
echo "  models:   $MODELS_JSON"
echo "  suite:    $SUITE_JSON"
echo "  repeats:  $REPEATS"
echo "  sandbox:  ${SANDBOX_BACKEND} (per-run=${SANDBOX_PER_RUN})"
echo "  seed:     ${SCENARIO_SEED}"
echo "  tools:    ${REQUEST_TOOLS}"
echo "  logprobs: ${REQUEST_LOGPROBS}"

SUITE_ID="$("$PY" -c "import json; suite=json.load(open('$SUITE_JSON')); print(suite.get('suite_id','suite'))")"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_ROOT="${EXPERIMENT_RESULTS_DIR:-data/results}"
DATASETS_ROOT="${EXPERIMENT_DATASETS_DIR:-data/datasets}"
mkdir -p "$RESULTS_ROOT" "$DATASETS_ROOT"
OUT_PATH="${RESULTS_ROOT}/benchmark_${SUITE_ID}_${TS}.csv"
DATASET_DIR="${DATASETS_ROOT}/benchmark_${SUITE_ID}_${TS}"

"$PY" experiments/run_benchmark.py \
  --models "$MODELS_JSON" \
  --suite "$SUITE_JSON" \
  --repeats "$REPEATS" \
  --probe-mode off \
  --out "$OUT_PATH"

echo ""
echo "Summary..."
"$PY" analysis/analyze_benchmark.py --results "$OUT_PATH" --out "${OUT_PATH%.csv}_summary.csv"

echo ""
echo "Shock deltas..."
"$PY" analysis/analyze_shock_deltas.py --results "$OUT_PATH" --out "${OUT_PATH%.csv}_shock_deltas.csv"

echo ""
echo "Package dataset..."
"$PY" analysis/package_benchmark.py --results "$OUT_PATH" --out-dir "$DATASET_DIR" --copy-logs --copy-manifests

echo ""
echo "Figures..."
"$PY" analysis/plot_paper_figures.py --results "$OUT_PATH" --out-dir "$DATASET_DIR/figures"

echo ""
echo "Done."
echo "Raw:     $OUT_PATH"
echo "Summary: ${OUT_PATH%.csv}_summary.csv"
echo "Deltas:  ${OUT_PATH%.csv}_shock_deltas.csv"
echo "Dataset:  $DATASET_DIR"
