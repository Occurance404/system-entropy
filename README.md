# Entropic Stress-Test Framework

A research framework to study how LLM-based agents behave under non-stationary tasks. It couples a controllable orchestrator, branching probes (Semantic Collapse Ratio/SCR), and a Docker sandbox to reveal silent failure modes and trigger rescue handoffs.

## Quick Start
- Requirements: Python 3.11+, `virtualenv`; Docker needed for TerminalBench sandbox runs.
- Install:
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Configure `.env` (example):
  ```ini
  VLLM_API_KEY=sk-...
  VLLM_BASE_URL=https://api.deepseek.com/v1
  VLLM_MODEL_NAME=deepseek-chat
  RESCUE_API_KEY=sk-...        # optional
  RESCUE_BASE_URL=https://api.openai.com/v1
  RESCUE_MODEL_NAME=gpt-4      # optional
  PROXY_AUTH_TOKEN=dev-secret  # for proxy routes
  ```

## Running Experiments
- Baseline or Rescue run (with perturbations, optional handoff):
  ```bash
  python experiments/run_rescue_experiment.py --scenario_id drug_filter_shock --max_steps 20            # baseline
  python experiments/run_rescue_experiment.py --scenario_id drug_filter_shock --max_steps 20 --enable_rescue
  ```
- Hard mode (no rescue, periodic probes):
  ```bash
  python experiments/run_hard_mode.py --scenario_id hard_coding_challenge --max_steps 20 --probe_interval 3
  ```
- Real simulation with a remote model (uses `.env`):
  ```bash
  python experiments/simulate_real.py --scenario_id drug_filter_shock --max_steps 10
  python experiments/simulate_real.py --scenario_id drug_filter_baseline --max_steps 50 --cheap  # disables probes + stops on validator success
  ```
- Smoke test (plumbing/logging sanity):
  ```bash
  python scripts/smoke_test.py
  ```
- Scale experiment (slow, aggregates logs):
  ```bash
  python experiments/run_scale_experiment.py --num_runs 5
  ```
- TerminalBench harness (requires Docker, proxy):
  ```bash
  ./scripts/run_tb_task.sh --task-id bank-trans-filter
  ```

Logs land under `logs/rescue/`, `logs/hard_mode/`, or `logs/terminal_bench/`; sandboxes live under `data/sandbox_<scenario>/`.
Per-run manifests and summaries land under `data/run_artifacts/<run_id>/`.

## Core Concepts
- **Orchestrator** (`src/orchestrator/engine.py`, implementation in `src/orchestrator/core/orchestrator.py`): injects perturbations, runs branching probes, tracks panic counters, and can switch to a rescue agent.
- **Agents**: `OpenAICompatibleAgent` (tool-calling, async probes) and `ScriptedAgent` (mock). Tools: read/write file, run shell, execute Python, stub web search.
- **Metrics** (`src/services/metrics.py`):
  - Entropy (chosen-token surprisal), IGE `(H_pre - H_post)/token_cost` around tool calls.
  - SCR (avg cosine distance of 5 probe branches via `all-MiniLM-L6-v2`; `None` if embeddings missing).
  - RDI (cosine distance to scenario ground truth), compression ratio, cyclomatic complexity.
- **Intervention logic**: entropy threshold 0.8 (or z-score >2), panic if 3 consecutive breaches; optional silent SCR probes via `probe_interval`.
- **Proxy** (`src/llm_proxy.py`): FastAPI shim to the real model with optional shock injection; logs via `TerminalBenchMonitor`.

## Scenarios
Defined in `src/scenarios/definitions.py` with setup ops in `src/scenarios/setup_ops.py`:
- Drug filter (baseline/shock), File organizer shock, Data pipeline shock, Vision defect shock, Hard coding/analysis challenges.

## Repo Structure
- `src/agent/` agents; `src/orchestrator/` control loop; `src/services/` metrics; `src/tools/` tool registry; `src/connectors/` Docker sandbox.
- `experiments/`: experiment runners; `analysis/`: plots + aggregation helpers; `scripts/`: ops utilities.
- `docs/metrics.md`, `paper/`, `thesis/`, `archive/legacy_docs/`: documentation/thesis material.
- `data/`: sandboxes, logs, results (ignored in git; generated).

## Testing
Run unit/integration tests:
```bash
pytest
```
Smoke test for logging:
```bash
python scripts/smoke_test.py
```

## Notes
- Generated artifacts (logs/results/sandboxes) are ignored via `.gitignore`.
- SCR/RDI use SentenceTransformers when available; if the embedding model is not cached locally, the framework falls back to a deterministic hashing embedder so runs stay offline-friendly.
- Control the embedding behavior with `SCR_EMBEDDING_BACKEND=auto|st|hash` and `SCR_LOCAL_FILES_ONLY=1|0` (default: `1`, no downloads).
- For production-style runs, set `PROXY_AUTH_TOKEN` and start `src/llm_proxy.py` before agents.
- If Docker is unavailable, set `SANDBOX_BACKEND=local` (or leave `auto` to fall back) to run sandboxes directly on your machine.
- For paper datasets, set `SANDBOX_PER_RUN=1` to prevent runs from overwriting the same `data/sandbox_<scenario>` directory.
- For deterministic inputs across runs, set `SCENARIO_SEED=0` (or vary it intentionally to measure robustness).
- If a provider rejects `logprobs`, set `REQUEST_LOGPROBS=off` (default `auto` disables logprobs after the first rejection).
- If a provider rejects OpenAI tool calling (`tools` / `tool_choice`), set `REQUEST_TOOLS=off` to use text-based JSON tool calls (default `auto` falls back after the first rejection).
- Optional: expose validator failures back to the agent with `VALIDATION_FEEDBACK=on` (default `off`).

### Local Ollama (Zero Credits)

Ollama exposes an OpenAI-compatible endpoint on `http://127.0.0.1:11434/v1`:

```bash
export VLLM_BASE_URL=http://127.0.0.1:11434/v1
export VLLM_API_KEY=ollama
export VLLM_MODEL_NAME=deepseek-r1:14b
export REQUEST_TOOLS=off
export REQUEST_LOGPROBS=off
export SANDBOX_BACKEND=local
```

## Safety Guards (V1)
- Tool outputs are truncated + redacted before being appended to the agent history (prevents context flooding and accidental secret exposure).
- `read_file` supports `"mode": "auto|full|outline"` and optional `start_line`/`end_line` for targeted expansion.
- `write_file` blocks likely secret material by default; configure with `SECRETS_POLICY=block|warn|off`.
