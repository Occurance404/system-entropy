#!/bin/bash

# Universal Agent Runner
# Usage: ./run_universal_agent.sh --agent-cmd "python my_agent.py" --task-id "drug_filter_shock"

export PYTHONPATH="$PYTHONPATH:$(pwd)"

# 1. Load Config
if [ -f .env ]; then
    export $(grep -v '^\s*#' .env | grep -v '^\s*$' | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

# Defaults
AGENT_CMD=""
TASK_ID="drug_filter_shock"

# Parse Args
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --agent-cmd) AGENT_CMD="$2"; shift ;;
        --task-id) TASK_ID="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$AGENT_CMD" ]; then
    echo "Error: You must provide an agent command with --agent-cmd"
    echo "Example: ./run_universal_agent.sh --agent-cmd 'python my_agent.py'"
    exit 1
fi

# 2. Start LLM Proxy (The Heart Rate Monitor)
PROXY_LOG="proxy_universal.log"
echo "[1/3] Starting LLM Proxy..."
nohup .venv/bin/python src/llm_proxy.py > "$PROXY_LOG" 2>&1 &
PROXY_PID=$!
echo "      Proxy PID: $PROXY_PID. Listening on http://localhost:8000"
sleep 10 # Increased wait time for slow startup

# 3. Start Environment (TerminalBench)
# We use 'tb server' (hypothetical) or a script to boot the container and keep it open
# For now, we will assume the Connector Script will handle the "Login" to the environment.
echo "[2/3] Preparing Environment (Task: $TASK_ID)..."
# Note: In a real generic setup, we'd spin up the Docker container here.
# For this prototype, we assume the agent script uses 'src/connectors/tb_connect.py'
# to find the active container or spawn one.

# 4. Run The Agent
echo "[3/3] Launching User Agent..."
echo "      Command: $AGENT_CMD"
echo "      Connecting Agent to Proxy..."

# Export Proxy Config for the Agent
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="${VLLM_API_KEY}" # Pass real key to proxy
export TB_TASK_ID="$TASK_ID"

# Add venv python to PATH for agent command
# This is usually for convenience, but for eval'ed commands, being explicit is safer.
# export PATH="$(pwd)/.venv/bin:$PATH" # Original line, now handled more robustly below

# Modify AGENT_CMD to explicitly use venv python if it's a python script
if [[ "$AGENT_CMD" == python* ]]; then
    AGENT_CMD="$(pwd)/.venv/bin/$AGENT_CMD"
fi

# Execute
eval "$AGENT_CMD"

# 5. Cleanup
echo "--- Task Finished ---"
echo "Stopping Proxy..."
kill "$PROXY_PID"
