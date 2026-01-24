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
export REQUEST_LOGPROBS="off"  # We still turn this off for Qwen/Ollama generally, 
                               # but SCR is computed via *embeddings* of the text, 
                               # not logprobs, so SCR works even if this is off.

export SANDBOX_BACKEND="local"
export SANDBOX_PYTHON="$VENV_PYTHON"

SCENARIO="hard_coding_challenge"

echo "----------------------------------------------------------------"
echo "STARTING PROBED RUN: $SCENARIO"
echo "Model: $LOCAL_MODEL | Probes: Every 5 steps"
echo "----------------------------------------------------------------"

# Clean sandbox
rm -rf "data/sandbox_$SCENARIO" 2>/dev/null

$VENV_PYTHON experiments/run_hard_mode.py \
    --scenario_id "$SCENARIO" \
    --max_steps 50 \
    --probe_interval 5

echo "Run Complete."
