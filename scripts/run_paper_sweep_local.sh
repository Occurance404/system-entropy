#!/usr/bin/env bash
set -euo pipefail

# One-command local sweep + summary.
#
# Usage:
#   ./scripts/run_paper_sweep_local.sh [models_json] [suite_json] [repeats]
#
# Defaults:
#   models_json = benchmarks/models.ollama.json
#   suite_json  = benchmarks/suite_v2.json
#   repeats     = 3
#
# Outputs:
#   - Raw results CSV: ${EXPERIMENT_RESULTS_DIR:-data/results}/benchmark_<suite>_<ts>.csv
#   - Summary CSV:     ${EXPERIMENT_RESULTS_DIR:-data/results}/benchmark_<suite>_<ts>_summary.csv

MODELS_JSON="${1:-benchmarks/models.ollama.json}"
SUITE_JSON="${2:-benchmarks/suite_v2.json}"
REPEATS="${3:-3}"

if [[ -x "./.venv/bin/python" ]]; then
  PY="./.venv/bin/python"
else
  PY="python3"
fi

# Local-first defaults (override by exporting the vars before running this script).
export SANDBOX_BACKEND="${SANDBOX_BACKEND:-local}"
export SANDBOX_PER_RUN="${SANDBOX_PER_RUN:-1}"
export REQUEST_TOOLS="${REQUEST_TOOLS:-off}"
export REQUEST_LOGPROBS="${REQUEST_LOGPROBS:-off}"
export SCR_LOCAL_FILES_ONLY="${SCR_LOCAL_FILES_ONLY:-1}"
export SCENARIO_SEED="${SCENARIO_SEED:-0}"

echo "Running benchmark sweep..."
echo "  python:   $PY"
echo "  models:   $MODELS_JSON"
echo "  suite:    $SUITE_JSON"
echo "  repeats:  $REPEATS"
echo "  sandbox:  ${SANDBOX_BACKEND}"
echo "  tools:    ${REQUEST_TOOLS}"
echo "  logprobs: ${REQUEST_LOGPROBS}"

RESULTS_ROOT="${EXPERIMENT_RESULTS_DIR:-data/results}"
mkdir -p "$RESULTS_ROOT"

OUT_PATH="$("$PY" -c "from datetime import datetime; import json; import os; suite=json.load(open('$SUITE_JSON')); sid=suite.get('suite_id','suite'); ts=datetime.now().strftime('%Y%m%d_%H%M%S'); print(os.path.join('$RESULTS_ROOT',f'benchmark_{sid}_{ts}.csv'))")"

"$PY" experiments/run_benchmark.py \
  --models "$MODELS_JSON" \
  --suite "$SUITE_JSON" \
  --repeats "$REPEATS" \
  --out "$OUT_PATH"

echo ""
echo "Aggregating summary..."
"$PY" analysis/analyze_benchmark.py --results "$OUT_PATH"

echo ""
echo "Done."
echo "Raw:     $OUT_PATH"
echo "Summary: ${OUT_PATH%.csv}_summary.csv"
