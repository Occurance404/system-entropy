#!/usr/bin/env bash
set -euo pipefail

if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

export SANDBOX_BACKEND="${SANDBOX_BACKEND:-local}"
export SANDBOX_PER_RUN="${SANDBOX_PER_RUN:-1}"
export SCENARIO_SEED="${SCENARIO_SEED:-0}"
export REQUEST_TOOLS="${REQUEST_TOOLS:-off}"
export REQUEST_LOGPROBS="${REQUEST_LOGPROBS:-off}"
export AI_VERIFIER="${AI_VERIFIER:-off}"

SCENARIOS=(
  archive_impossible_dates
  museum_renamed_species
  dream_court_transcript
  lunar_cargo_ritual
  paradox_lab_protocol
  oracle_contract_amendment
  city_duplicate_identities
  signal_mirror_logs
  missing_axiom
  bureau_contradictory_forms
)

echo "Running surreal offline smoke (scripted agent, no API spend)..."
for s in "${SCENARIOS[@]}"; do
  echo "--- $s ---"
  "$PY" experiments/simulate.py --scenario "$s" --steps 6 --cheap
  echo
 done

echo "Offline smoke completed."
