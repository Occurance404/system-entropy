# Surreal Cold-Start Runbook

This runbook covers the 10 new surreal scenarios integrated into the harness.

## What You Need To Download

1. Python dependencies once:
- `pip install -r requirements.txt` (or use project `.venv` if already set up)

2. Model access (for real-agent runs):
- No dataset download required.
- You only need API credentials/endpoints (for example OpenRouter key + model ID).

3. Optional embedding model download:
- If you want semantic SCR embeddings, `SentenceTransformer` may download `all-MiniLM-L6-v2` once.
- If you want strict offline mode, set:
  - `SCR_LOCAL_FILES_ONLY=1`
  - `SCR_EMBEDDING_BACKEND=hash`

## Scenarios: Data Download Requirement

All scenarios below are fully local and run from repository files under `scenarios/surreal_v1/*/inputs`.

- `archive_impossible_dates`: no data download
- `museum_renamed_species`: no data download
- `dream_court_transcript`: no data download
- `lunar_cargo_ritual`: no data download
- `paradox_lab_protocol`: no data download
- `oracle_contract_amendment`: no data download
- `city_duplicate_identities`: no data download
- `signal_mirror_logs`: no data download
- `missing_axiom`: no data download
- `bureau_contradictory_forms`: no data download

## Cold-Start Commands (Offline Sanity)

Use scripted agent + local sandbox, zero API spend:

```bash
export SANDBOX_BACKEND=local
export SCENARIO_SEED=0
export REQUEST_TOOLS=off
export REQUEST_LOGPROBS=off
export AI_VERIFIER=off

.venv/bin/python experiments/simulate.py --scenario archive_impossible_dates --steps 6 --cheap
```

## Real-Agent Commands (API Spend)

```bash
export SANDBOX_BACKEND=local
export SANDBOX_PER_RUN=1
export SCENARIO_SEED=0

# model endpoint/key envs
export VLLM_BASE_URL="https://openrouter.ai/api/v1"
export VLLM_API_KEY="<your_key>"
export VLLM_MODEL_NAME="<provider/model>"

.venv/bin/python scripts/run_fair_real_session.py \
  --mode rescue_baseline \
  --scenario_id archive_impossible_dates \
  --max_steps 40
```

## Suite File

Use:
- `benchmarks/suite_surreal_v1.json`

