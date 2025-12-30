# Paper Runsheet (Reproduce Tables + Figures)

This is the minimal command set to regenerate the paper’s benchmark tables and figures.

## 0) One-time setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Zero-credit local runs (Ollama)

Ollama exposes an OpenAI-compatible endpoint at `http://127.0.0.1:11434/v1`.

```bash
export VLLM_BASE_URL=http://127.0.0.1:11434/v1
export VLLM_API_KEY=ollama

# For Ollama models, disable features that commonly cause request rejection.
export REQUEST_TOOLS=off
export REQUEST_LOGPROBS=off
export SANDBOX_BACKEND=local
```

Use the provided example model list:

```bash
.venv/bin/python check_model_capabilities.py --models benchmarks/models.ollama.example.json
```

## 2) Benchmark sweep (raw CSV)

Recommended (paper-style): shock-only probes, small branch count.

```bash
.venv/bin/python run_benchmark.py \
  --models benchmarks/models.ollama.example.json \
  --suite benchmarks/suite_v2.json \
  --repeats 3 \
  --probe-mode shock \
  --probe-branches 3 \
  --max-steps 60
```

Output lands in `data/results/benchmark_<suite>_<timestamp>.csv`.

## 3) Aggregate into paper tables (summary CSV)

```bash
.venv/bin/python analyze_benchmark.py \
  --results data/results/benchmark_<suite>_<timestamp>.csv
```

This writes `data/results/benchmark_<...>_summary.csv` with:
- `success_rate`, `median_steps`
- `median_total_tokens`, `median_probe_tokens`, `median_total_tokens_incl_probes`
- `mean_entropy_coverage` (fraction of steps with entropy present; excludes probes)
- `median_scr_shock` (peak SCR at perturbations, per run → median)

## 4) Figures

If you have a representative run log (`.jsonl`), generate the multi-panel “experiment summary” plot:

```bash
.venv/bin/python visualize_results.py --log_file data/logs_terminal_bench/<run>.jsonl
```

By default figures land under `data/results/`.

## 5) What to paste into the paper

- Table: `data/results/benchmark_<...>_summary.csv`
- Figure: `data/results/experiment_summary.png` (from a representative run)
