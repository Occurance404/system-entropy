# How to Run the Entropic Stress-Test Experiments

This guide explains how to run the "Cognitive Collapse" experiments, including the baseline (failure observation) and the rescue protocol (model handoff).

## 1. Prerequisites

Ensure you have the virtual environment set up and dependencies installed:

```bash
# Create a venv if you don't have one yet
python3 -m venv .venv

# Activate the virtual environment (if not already active)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
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

### No-Block Toggles (Recommended)

If you want the framework to “just run” across providers with different feature support:

```bash
# Avoid Docker bottlenecks (runs sandboxes on your machine).
export SANDBOX_BACKEND=local   # or "auto" (default) to fall back automatically

# Avoid logprob bottlenecks (entropy becomes optional).
export REQUEST_LOGPROBS=auto   # auto|off|on

# Avoid tool-calling bottlenecks (falls back to text-based JSON tool calls).
export REQUEST_TOOLS=auto      # auto|off|on

# Optional: feed validator failures back into the agent context (can improve completion rates).
export VALIDATION_FEEDBACK=off # off|on

# Optional: do not block writes that look like secrets (be careful).
export SECRETS_POLICY=warn     # block|warn|off
```

### Local Ollama (Zero Credits)

If you’re running a local model via Ollama (port `11434`), set:

```bash
export VLLM_BASE_URL=http://127.0.0.1:11434/v1
export VLLM_API_KEY=ollama
export VLLM_MODEL_NAME=deepseek-r1:14b

# Recommended for Ollama models that reject native `tools`:
export REQUEST_TOOLS=off
export REQUEST_LOGPROBS=off
export SANDBOX_BACKEND=local
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
For orchestrator-based runs that use the unified monitor, per-run artifacts (manifest + summary) are saved under `data/run_artifacts/<run_id>/`.

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

## 5. Cheap Mode (Low-Credit Runs)

If you are low on API credits, prefer runs that disable expensive branching probes and stop early when the validator reports success:

```bash
python run_hard_mode.py --scenario_id drug_filter_baseline --max_steps 50 --cheap
python simulate_real.py --scenario_id drug_filter_baseline --max_steps 50 --cheap
```

## 6. Benchmark Sweeps (Paper Tables)

For multi-model sweeps with deterministic validators, use `run_benchmark.py` + `analyze_benchmark.py`.

1) Create a models file:
- Start from `benchmarks/models.paper.template.json` and replace the model IDs.
- Put keys in `.env` (recommended): `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, etc.
 - Ensure Docker is running and your user can access it (TerminalBench sandbox uses Docker).

2) (Optional) Probe provider capabilities (logprobs + tools):
```bash
python3 check_model_capabilities.py --models benchmarks/models.paper.template.json
python3 check_model_capabilities.py --models benchmarks/models.ollama.example.json
```

3) Run a small sweep (cheap-ish defaults):
```bash
python run_benchmark.py \
  --models benchmarks/models.paper.template.json \
  --suite benchmarks/suite_v1.json \
  --repeats 1 \
  --probe-mode shock \
  --probe-branches 3
```

4) Aggregate into a summary CSV:
```bash
python analyze_benchmark.py --results data/results/benchmark_v1_<timestamp>.csv
```
