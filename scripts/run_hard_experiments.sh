#!/bin/bash

VENV_PYTHON="$(pwd)/.venv/bin/python"

# Force Local Config
LOCAL_URL="http://127.0.0.1:11434/v1"
LOCAL_KEY="ollama"
LOCAL_MODEL="qwen3:14b"

export VLLM_BASE_URL="$LOCAL_URL"
export VLLM_API_KEY="$LOCAL_KEY"
export VLLM_MODEL_NAME="$LOCAL_MODEL"
export REQUEST_TOOLS="off"
export REQUEST_LOGPROBS="off"
export SANDBOX_BACKEND="local"
export SANDBOX_PYTHON="$VENV_PYTHON"

# STRICT DATA COLLECTION CONFIGURATION
MAX_STEPS=100         # Enough to capture the "rot" / loop
PROBE_INTERVAL=3      # High resolution SCR (every 3 steps)
PROBE_BRANCHES=3      # 3 futures per probe

SCENARIOS=(
    "hard_coding_challenge"
    "hard_analysis_challenge"
    "legacy_refactor_challenge"
    "hard_socket_challenge"
    "dirty_data_challenge"
    "full_stack_challenge"
    "startup_acquisition_challenge"
)

echo "Starting THESIS DATA COLLECTION (High Resolution)"
echo "Model: $LOCAL_MODEL | Steps: $MAX_STEPS | Probes: Every $PROBE_INTERVAL steps"

for SCENARIO in "${SCENARIOS[@]}"; do
    echo "================================================================"
    echo "SCENARIO: $SCENARIO"
    echo "================================================================"
    
    # Attempt cleanup
    rm -rf "data/sandbox_$SCENARIO" 2>/dev/null
    
    $VENV_PYTHON experiments/run_hard_mode.py \
        --scenario_id "$SCENARIO" \
        --max_steps "$MAX_STEPS" \
        --probe_interval "$PROBE_INTERVAL" \
        --autonomous
    
    echo "Finished $SCENARIO"
    sleep 5
done

echo "SEQUENCE COMPLETE."
