# How to Run the Entropic Stress-Test Experiments

This guide explains how to run the "Cognitive Collapse" experiments, including the baseline (failure observation) and the rescue protocol (model handoff).

## 1. Prerequisites

Ensure you have the virtual environment set up and dependencies installed:

```bash
# Activate the virtual environment (if not already active)
source .venv/bin/activate
```

## 2. Configuration

Make sure your `.env` file is configured with your model provider credentials.

**Example `.env`:**
```ini
# Primary Agent (The one being tested)
VLLM_API_KEY="sk-..."
VLLM_BASE_URL="https://api.deepseek.com/v1" 
VLLM_MODEL_NAME="deepseek-chat"

# Rescue Agent (The smarter model that steps in)
RESCUE_API_KEY="sk-..."
RESCUE_BASE_URL="https://api.openai.com/v1"
RESCUE_MODEL_NAME="gpt-4"
```

## 3. Running Experiments

We support two modes: **Baseline** (Observation) and **Rescue** (Intervention).

### Mode A: Baseline (No Rescue)
Run this to observe how the Primary Agent behaves under stress without interference. This establishes your "Control" group.

```bash
python3 run_rescue_experiment.py \
    --scenario_id drug_filter_shock \
    --max_steps 20
```
*Outcome:* The agent should likely hit the "Panic" threshold and eventually fail or get stuck in a loop.

### Mode B: Rescue Protocol (Intervention)
Run this to test the Handoff Mechanism. When "Panic" is detected, the system will automatically switch to the Rescue Agent.

```bash
python3 run_rescue_experiment.py \
    --scenario_id drug_filter_shock \
    --max_steps 20 \
    --enable_rescue
```
*Outcome:* When the `intervention` event triggers, the log will show `>>> INITIATING RESCUE PROTOCOL` and switch the model.

## 4. Viewing Results

Logs are saved to `data/logs_rescue/`.

Each log file is a JSONL file containing step-by-step metrics:
- **SCR (Semantic Collapse Ratio):** Measures confusion.
- **RDI (Regressive Debt Index):** Measures goal drift.
- **Panic Counter:** Tracks consecutive high-entropy states.

**To analyze a run:**
You can use `grep` to quickly see critical events:

```bash
# See when the perturbation happened
grep "perturbation_triggered" data/logs_rescue/latest_log.jsonl

# See when panic triggered an intervention
grep "intervention" data/logs_rescue/latest_log.jsonl
```
