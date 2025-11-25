#!/bin/bash

# Ensure the current directory is in PYTHONPATH for module imports
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Load environment variables from .env file, ignoring comments and empty lines
if [ -f .env ]; then
    export $(grep -v '^\s*#' .env | grep -v '^\s*$' | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

# Map VLLM_ variables to standard OpenAI/LiteLLM variables for TerminalBench
# Use a custom env var that LiteLLM will pick up inside the Docker container
export LLM_PROXY_BASE_URL="http://host.docker.internal:8000/v1" # Docker-specific host address
export OPENAI_API_BASE="http://localhost:8000/v1" # Host-local address for the agent running on host
export OPENAI_API_KEY="${VLLM_API_KEY}" # Still pass the real API key to the proxy

# Default model name if not explicitly set in .env or for litellm's sake
# We'll use the VLLM_MODEL_NAME directly as the proxy will pass it through to the real LLM.
# Prefix with openai/ to ensure LiteLLM uses the generic OpenAI provider logic (respecting API_BASE)
MODEL_TO_USE="openai/${VLLM_MODEL_NAME}"

# Ensure MODEL_TO_USE is not empty
if [ -z "$MODEL_TO_USE" ]; then
    echo "Error: VLLM_MODEL_NAME not set in .env file."
    exit 1
fi

# Default arguments for tb run
TB_DATASET_PATH="terminal-bench/tasks"
TB_TASK_ID="drug_filter_shock" # Default task for initial test
TB_AGENT="mini-swe-agent" # Uses LiteLLM which we patched
TB_MODEL_ARG="--model $MODEL_TO_USE"

# Parse command line arguments to override defaults
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --task-id) TB_TASK_ID="$2"; shift ;;
        --model) TB_MODEL_ARG="--model $2"; shift ;;
        --agent) TB_AGENT="$2"; shift ;;
        --help) echo "Usage: ./run_tb_task.sh [--task-id <id>] [--model <model>] [--agent <agent>]"; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Running TerminalBench with:"
echo "  Task ID: $TB_TASK_ID"
echo "  Agent: $TB_AGENT"
echo "  Model: $MODEL_TO_USE"
echo "  OpenAI API Base: $OPENAI_API_BASE"
env | grep OPENAI

# --- Start LLM Proxy in background ---
PROXY_LOG_FILE="proxy.log"
echo "Starting LLM proxy server in background. Logs will be in $PROXY_LOG_FILE"
# Ensure PYTHONPATH is set for the proxy process directly in the nohup command using env
nohup /usr/bin/env PYTHONPATH="$(pwd)${PYTHONPATH:+:}$PYTHONPATH" .venv/bin/python src/llm_proxy.py > "$PROXY_LOG_FILE" 2>&1 &
PROXY_PID=$!
echo "LLM Proxy PID: $PROXY_PID"
sleep 5 # Give proxy time to start up

# --- Execute TerminalBench ---
echo "Running TerminalBench tasks. Output will be visible directly in console."
.venv/bin/tb run \
    --dataset-path "$TB_DATASET_PATH" \
    --task-id "$TB_TASK_ID" \
    --agent "$TB_AGENT" \
    $TB_MODEL_ARG \
    --no-cleanup \
    --log-level debug \
    --livestream \
    --global-agent-timeout-sec 600 \
    --global-test-timeout-sec 600 \
    --n-concurrent 1 \
    --n-attempts 1

# --- Stop LLM Proxy ---
echo "Stopping LLM proxy (PID: $PROXY_PID)..."
kill "$PROXY_PID"
wait "$PROXY_PID" 2>/dev/null # Wait for it to terminate
echo "LLM Proxy stopped."

echo "TerminalBench run complete. Check $TB_OUTPUT_LOG for harness output and $PROXY_LOG_FILE for proxy logs."
