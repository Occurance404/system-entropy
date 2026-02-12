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

# Optional: keep each run's sandbox output (prevents overwriting).
# Recommended for paper datasets.
export SANDBOX_PER_RUN=1

# Optional: python interpreter used by the `execute_python` tool.
# Local sandbox defaults to repo `.venv` when available; override here if needed.
export SANDBOX_PYTHON=python3

# Optional: make scenario inputs deterministic across runs (recommended for benchmarks).
export SCENARIO_SEED=0

# Avoid logprob bottlenecks (entropy becomes optional).
export REQUEST_LOGPROBS=auto   # auto|off|on

# Avoid tool-calling bottlenecks (falls back to text-based JSON tool calls).
export REQUEST_TOOLS=auto      # auto|off|on

# Optional: feed validator failures back into the agent context (can improve completion rates).
export VALIDATION_FEEDBACK=off # off|on

# Token caps (recommended for local/open models to avoid huge generations)
export MAX_COMPLETION_TOKENS=1024
export PROBE_MAX_TOKENS=192

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

### Quick sanity check (recommended)

```bash
python3 scripts/check_setup.py
```

This reports whether Docker is usable (optional), whether your model endpoint is reachable, and which SCR embedding backend is active (semantic vs offline hashing fallback).

### Managed Session Runner (strongly recommended)

To avoid mixed outputs, run experiments through the session wrapper. Each run gets its own folder with:
- isolated logs
- isolated run artifacts/manifests
- isolated result CSVs
- command + metadata in `metadata/session.json`

```bash
.venv/bin/python scripts/run_experiment_session.py \
  --name rescue_drug_filter_real \
  --notes "real model baseline run" \
  -- \
  .venv/bin/python experiments/run_rescue_experiment.py \
    --scenario_id drug_filter_shock \
    --max_steps 20
```

Session outputs are written to:
`data/experiments/<timestamp>_<name>/`

### Fair Real-Agent Runner (recommended for clean intelligence evaluation)

Use this when you want to reduce technical noise (tool capability mismatch, probe overhead, mixed logs) and give the agent a fair chance:

```bash
.venv/bin/python scripts/run_fair_real_session.py \
  --mode rescue_baseline \
  --scenario_id drug_filter_shock \
  --max_steps 20
```

What it does automatically:
- preflights endpoint reachability
- probes model support for tools/logprobs and sets `REQUEST_TOOLS` / `REQUEST_LOGPROBS`
- forces stable defaults (`SANDBOX_BACKEND=local`, `SANDBOX_PER_RUN=1`, `SCENARIO_SEED=0`)
- runs through the managed session wrapper for isolated artifacts

Modes:
- `rescue_baseline` (default): real agent, no rescue handoff
- `rescue`: enables rescue handoff
- `hard`: hard-mode runner with `--require_real_agent`
- `simulate`: runs `experiments/simulate_real.py`

### Mode A: Baseline (No Rescue)
Run this to observe how the Primary Agent behaves under stress without interference. This establishes your "Control" group.

```bash
python3 experiments/run_rescue_experiment.py \
    --scenario_id drug_filter_shock \
    --max_steps 20
```
*Outcome:* The agent should likely hit the "Panic" threshold and eventually fail or get stuck in a loop.

### Mode B: Rescue Protocol (Intervention)
Run this to test the Handoff Mechanism. When "Panic" is detected, the system will automatically switch to the Rescue Agent.

```bash
python3 experiments/run_rescue_experiment.py \
    --scenario_id drug_filter_shock \
    --max_steps 20 \
    --enable_rescue
```
*Outcome:* When the `intervention` event triggers, the log will show `>>> INITIATING RESCUE PROTOCOL` and switch the model.

## 4. Viewing Results

Logs are saved to `logs/rescue/`.
For orchestrator-based runs that use the unified monitor, per-run artifacts (manifest + summary) are saved under `data/run_artifacts/<run_id>/`.
If you use the managed session runner, these paths are isolated under:
`data/experiments/<timestamp>_<name>/logs/...` and `data/experiments/<timestamp>_<name>/run_artifacts/...`.

Each log file is a JSONL file containing step-by-step metrics:
- **SCR (Semantic Collapse Ratio):** Measures confusion.
- **RDI (Regressive Debt Index):** Measures goal drift.
- **Panic Counter:** Tracks consecutive high-entropy states.

**To analyze a run:**
You can use `grep` to quickly see critical events:

```bash
# See when the perturbation happened
grep "perturbation_triggered" logs/rescue/latest_log.jsonl

# See when panic triggered an intervention
grep "intervention" logs/rescue/latest_log.jsonl
```

## 5. Cheap Mode (Low-Credit Runs)

If you are low on API credits, prefer runs that disable expensive branching probes and stop early when the validator reports success:

```bash
python experiments/run_hard_mode.py --scenario_id drug_filter_baseline --max_steps 50 --cheap
python experiments/simulate_real.py --scenario_id drug_filter_baseline --max_steps 50 --cheap
```

## 6. Benchmark Sweeps (Paper Tables)

For multi-model sweeps with deterministic validators, use `experiments/run_benchmark.py` + `analysis/analyze_benchmark.py`.

1) Create a models file:
- Start from `benchmarks/models.paper.template.json` and replace the model IDs.
- Put keys in `.env` (recommended): `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, etc.
 - Ensure Docker is running and your user can access it (TerminalBench sandbox uses Docker).

### OpenRouter paper sweep (no Docker)

If you’re using OpenRouter and want an end-to-end “paper sweep” (baseline vs shock) with deterministic validators:

1) Create `benchmarks/models.openrouter.paper.json` from `benchmarks/models.openrouter.paper.template.json`.
2) Set your key:
```bash
export OPENROUTER_API_KEY="..."
```
3) Run:
```bash
./scripts/run_paper_sweep_openrouter.sh benchmarks/models.openrouter.paper.json benchmarks/suite_paper_v1.json 10
```

Outputs:
- Raw CSV: `data/results/benchmark_<suite>_<ts>.csv`
- Packaged dataset: `data/datasets/benchmark_<suite>_<ts>/` (includes logs, manifests, delta table, figures)

2) (Optional) Probe provider capabilities (logprobs + tools):
```bash
python3 scripts/check_model_capabilities.py --models benchmarks/models.paper.template.json
python3 scripts/check_model_capabilities.py --models benchmarks/models.ollama.example.json
```

3) Run a small sweep (cheap-ish defaults):
```bash
python experiments/run_benchmark.py \
  --models benchmarks/models.paper.template.json \
  --suite benchmarks/suite_v1.json \
  --repeats 1 \
  --probe-mode shock \
  --probe-branches 3
```

4) Aggregate into a summary CSV:
```bash
python analysis/analyze_benchmark.py --results data/results/benchmark_v1_<timestamp>.csv
```

### One-command local sweep (no Docker)

If you're using a local OpenAI-compatible endpoint (e.g., Ollama) and want a single command that runs a sweep + summary:

```bash
./scripts/run_paper_sweep_local.sh benchmarks/models.ollama.json benchmarks/suite_v2.json 3
```
