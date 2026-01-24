# Entropic Stress-Testing of Tool-Using LLM Agents Under Non-Stationary Task Constraints

**Authors:** [Your Name], [Supervisor/Co-authors]  
**Affiliation:** [University / Lab]  
**Target:** arXiv + [venue TBD]  
**Status:** Draft (needs final benchmark runs)

## Abstract
Current evaluation frameworks for tool-using Large Language Model (LLM) agents largely measure success on fixed objectives (static solvability). In real deployments, objectives can shift mid-execution (non-stationary constraints), and agents can silently fail by persisting with obsolete plans, repeating actions, or “hallucinating progress”. We introduce an **Entropic Stress-Test** evaluation harness that injects controlled requirement changes (“shocks”) into deterministic sandboxes and measures agent stability and adaptation. We propose **Semantic Collapse Ratio (SCR)**, computed via branching probes that sample multiple possible next steps and quantify their semantic divergence, and **Information Gain Efficiency (IGE)**, a token-cost-normalized uncertainty reduction measure when token logprobs are available. A key practical finding is that logprob-based confidence is frequently unavailable or inconsistent across providers/models; treating missing logprobs as “0.0 entropy” is invalid. We show how SCR plus outcome-based validators provide a robust evaluation signal even when token-level confidence is missing, enabling reproducible comparisons across heterogeneous model providers.

## 1. Introduction
The rapid adoption of LLM-based agents for autonomous software engineering has highlighted a critical reliability gap. While agents perform well on static benchmarks, their performance degrades significantly in long-horizon tasks with changing requirements. This degradation is often characterized not by immediate failure, but by a gradual "Context Rot," where the agent's internal state drifts from the ground truth, leading to repetitive loops and hallucinated progress.

Standard observability metrics, such as token-level entropy or perplexity proxies, are unreliable in practice: many APIs do not expose token logprobs, and models optimized for helpfulness can produce confident language even when plans are brittle. This creates an observability gap for agent operators: failures may be silent until an external validator fails (or a human notices).

This paper contributes:
1. **A reproducible non-stationary evaluation harness** (orchestrator + deterministic sandboxes + validators) for tool-using agents under controlled requirement shocks.
2. **A provider-agnostic stability probe (SCR)**: an active measure of plan instability under shocks via semantic divergence across branched next-step generations.
3. **A practical observability result:** entropy proxies derived from token logprobs are often unavailable across providers/models; we therefore report **entropy coverage** and treat missing entropy as `null`, not `0.0`.
4. **A secondary behavioral result:** even when entropy is available, it can remain flat/low around shocks while agent behavior destabilizes, whereas SCR spikes near perturbations.

## 2. Methodology

### 2.1 System Architecture and Agent Loop
The Entropic Stress-Test framework has four decoupled components:
* **Orchestrator:** runs the step loop, injects perturbations, and coordinates probes (`src/orchestrator/core/orchestrator.py`).
* **Sandbox:** deterministic filesystem and command execution (Docker optional with local fallback).
* **Agent:** produces either a tool action or a natural-language reply at each step (`src/agent/real_agent.py`).
* **Monitor + Validators:** per-step logging and deterministic success checks (`src/monitor/terminal_bench_monitor.py`, `src/scenarios/validation_ops.py`).

Each step emits a structured JSONL record with scenario ID, step index, event type, tool call (if any), validator status, token usage, and metrics. Logs are written under `logs/terminal_bench/`.

### 2.2 Scenario Design and Non-Stationarity
Each scenario specifies: (1) an initial prompt, (2) a schedule of perturbations (step index plus new instruction), and (3) a validator that checks artifacts in the sandbox. Perturbations are injected as explicit user messages and can optionally mutate sandbox state. This yields deterministic tasks with controlled requirement changes.

The benchmark suite used for paper tables is defined in `benchmarks/suite_v2.json` and executed with `experiments/run_benchmark.py`.

### 2.3 Action Protocol and Token Accounting
At each step, the agent returns either:
* `tool_use`: a tool name plus JSON arguments executed in the sandbox, or
* `llm_reply`: a natural-language response (used on completion or failure).

Token usage is recorded when the provider exposes it (`prompt_tokens`, `completion_tokens`, `total_tokens`). Probe calls are tracked separately as `probe_total_tokens` to report totals with and without probes.

### 2.4 Branching Probe Protocol
To measure hidden uncertainty, we run branching probes at perturbation steps and optionally at periodic intervals.
* Shock probes: always run at perturbation steps.
* Periodic probes: configurable interval (default 5) when no perturbation is triggered.
* Branch count: N branches (default 3).

Each probe forks the current dialogue state (plus the perturbation message if present) and generates N independent "next step" thoughts using a higher temperature (0.9) to encourage divergence. The probe does not execute tools; it only samples candidate next actions for analysis.

### 2.5 Metrics
We report stability and observability metrics that do not depend on a specific provider:

**Semantic Collapse Ratio (SCR).** We compute SCR as the mean pairwise cosine distance between embedding vectors of the N probe branches:

$$ SCR = \frac{1}{N(N-1)} \sum_{i \ne j} d_{cos}(v_i, v_j) $$

Where $d_{cos}$ is cosine distance and $v_i$ are sentence embeddings from a SentenceTransformer model (default `all-MiniLM-L6-v2`). If embeddings are unavailable, SCR is recorded as null and excluded from summaries.

**Entropy coverage.** Token logprobs are optional across providers, so we compute entropy only when logprobs are present. We report entropy coverage: the fraction of non-probe steps with entropy values available.

**Entropy definition (chosen-token proxy).** When logprobs are available, we define entropy as the average negative log probability of the chosen tokens:

$$ H = -\frac{1}{N} \sum_{i=1}^{N} \log p(t_i) $$

This is a chosen-token surprisal proxy (not full distributional entropy), because most providers do not expose the full token distribution. We therefore always report coverage and treat missing values as null.

**Information Gain Efficiency (IGE).** When logprobs exist, we compute:

$$ IGE = \frac{H_{pre} - H_{post}}{TokenCost} $$

where $H$ is the average negative log probability of chosen tokens around a tool action. IGE is undefined when logprobs are absent.

### 2.6 Evaluation Protocol
Each (model, scenario) pair is run for a fixed maximum step budget (default 60 unless overridden by the suite). We repeat each run multiple times (default 3) to estimate variance. Success is determined solely by deterministic validators.

Aggregate metrics are computed with `analysis/analyze_benchmark.py` and include success rate, median steps, median token usage, mean entropy coverage, and median peak SCR at perturbation steps.

## 3. Experiments

### 3.1 Setup: The "Drug Filter Shock" Scenario
We designed a scenario simulating a real-world data engineering task:
1.  **Initial Goal:** Filter `drugs.csv` by `weight < 150` and write `filtered_by_weight.csv`.
2.  **Phase 1 (Steps 1-3):** Agent writes code and generates the CSV.
3.  **Shock 1 (Step 4):** A "Manager" injects a constraint change: the weight filter must now use an external `get_molecular_mass(drug_name)` API.
4.  **Shock 2 (Step 7):** The primary filter reverts back to the `weight` column, but the molecular mass API connection must remain present in code for future use.

### 3.2 The "Rescue" Protocol
We include an optional intervention protocol. After persistent “panic” (entropy-based when available, loop-based otherwise), the controller can: (a) inject a corrective system message, or (b) in a two-model setup, hand off control to a stronger “rescue” model. This protocol is evaluated separately from baseline runs to avoid conflating detection and recovery.

### 3.3 Evaluation Outputs (Paper Tables)
We report:
* **Success rate** (validator pass rate) per (model, scenario)
* **Median steps** to completion / termination
* **Token usage** (including optional probe tokens)
* **Entropy coverage** (fraction of steps with entropy present)
* **Peak SCR at shocks** (median peak SCR over perturbation events)

These aggregates are produced by:
* `experiments/run_benchmark.py` → raw results CSV
* `analysis/analyze_benchmark.py` → paper-ready summary CSV

Reproducibility runsheet: `paper/PAPER_RUNSHEET.md`.

## 4. Results

### 4.1 Primary Result: Entropy Coverage Makes “Entropy-Only” Observability Non-Viable
Across providers/models, token logprobs may be missing or inconsistent. As a result, entropy proxies cannot be relied upon as the sole observability signal in multi-provider sweeps. We treat entropy as an optional signal and report **entropy coverage**: the fraction of non-probe steps with entropy present.

**Key implication:** any evaluation that compares “entropy” across models without reporting coverage is at risk of silently mixing real values with missingness artifacts (e.g., treating missingness as `0.0`). [CITE]

### 4.2 Secondary Result: Entropy Can Decouple from Stability Even When Available
When token logprobs are available, chosen-token surprisal proxies may remain flat/low around perturbations even as agent behavior becomes unstable (e.g., repeating obsolete plans, oscillating between incompatible actions). In contrast, SCR is designed to measure plan instability directly by quantifying divergence across sampled next steps.

We therefore use SCR as the primary instability indicator around shocks, and entropy (when present) as a complementary, model-specific signal rather than a universal baseline. [CITE]

### 4.3 Metric Comparison (Qualitative + Aggregate)
We analyze metric behavior around perturbation steps. SCR spikes at shocks in scenarios where the agent must revise plans to satisfy new requirements. We report peak SCR at perturbations (median over repeats) as a compact instability summary statistic.

**TODO (after final runs):** include correlation plots between (SCR at shock, success rate) and (entropy proxy, success rate), and report confidence intervals across repeats.

![Figure 1: Experiment Summary showing the decorrelation between Entropy and SCR.](data/results/experiment_summary.png)
*Figure 1: Experiment Summary. Top to Bottom: (1) Entropy (flatline), (2) Panic Counter, (3) SCR (spikes at shock), (4) IGE, (5) Code Complexity, (6) Compression Ratio.*

### 4.4 Current Evidence (Existing Logs, No New Runs)
Because new experiments are currently blocked, we summarize **existing logs** only. We scanned all available JSONL logs and aggregated metadata into `data/results/existing_runs_summary.csv` (N=71 runs). Of these, **21 runs include SCR probes** and **50 runs are no-probe** (SCR unavailable).

**Probed runs (SCR available):**

| Scenario | Steps | Peak SCR | Entropy Coverage | Log |
|---|---:|---:|---:|---|
| data_pipeline_shock | 60 | 0.124 | 0.54 | tb_monitor_20260105_231033_997120.jsonl |
| drug_filter_shock | 60 | 0.116 | 0.57 | tb_monitor_20260105_223229_992630.jsonl |
| hard_socket_challenge | 50 | 0.790 | 1.00 | tb_monitor_20251220_155717.jsonl |
| dirty_data_challenge | 43 | 0.789 | 1.00 | tb_monitor_20251221_113918.jsonl |
| startup_acquisition_challenge | 41 | 0.714 | 1.00 | tb_monitor_20251221_170148.jsonl |
| legacy_refactor_challenge | 25 | 0.721 | 1.00 | tb_monitor_20251218_230512.jsonl |
| hard_coding_challenge | 24 | 0.636 | 0.00 | tb_monitor_20260106_092340_209163.jsonl |

**Long no-probe runs (SCR unavailable, loop signals visible):**

| Scenario | Steps | Min Compression | Entropy Coverage | Log |
|---|---:|---:|---:|---|
| full_stack_challenge | 200 | 0.026 | 0.00 | tb_monitor_20260106_050534_094984.jsonl |
| dirty_data_challenge | 200 | 0.013 | 0.00 | tb_monitor_20260106_032337_528833.jsonl |
| hard_socket_challenge | 200 | 0.090 | 0.00 | tb_monitor_20260106_022602_395608.jsonl |
| hard_analysis_challenge | 200 | 0.044 | 0.00 | tb_monitor_20260106_013357_268573.jsonl |
| hard_coding_challenge | 200 | 0.018 | 0.00 | tb_monitor_20260105_234746_694796.jsonl |
| data_pipeline_shock | 60 | 0.369 | 0.00 | tb_monitor_20251231_135534_396962.jsonl |
| drug_filter_shock | 60 | 0.357 | 0.00 | tb_monitor_20251231_130200_286313.jsonl |

**Interpretation (preliminary):**
* SCR spikes appear in probed runs around perturbations, while entropy coverage varies widely (0.00 to 1.00).
* In long no-probe runs, compression ratio collapses (<0.2) and stays low, indicating severe looping even when entropy is unavailable.
* This supports the claim that stability diagnostics must remain informative without logprobs, and that SCR (when available) complements loop-based signals.

## 5. Related Work
We position this work relative to: (1) tool-using agent benchmarks, (2) long-horizon context management, and (3) agent failure taxonomies. Our focus is not a new agent policy, but a reproducible *stress-test* harness for non-stationary objectives plus robust observability metrics when common confidence signals are missing.

**TODO:** finalize citations from `docs/references.bib` and add canonical references for CoT/ReAct/agent benchmarks used in the experimental design.

## 6. Discussion & Future Work
Active probing introduces costs and rate-limit pressure; our implementation supports configurable probe modes (off / shock-only / periodic) to study this trade-off. A practical limitation is that branching probes can stress free-tier API limits and require careful throughput planning.

**Future Work includes:**
1. **Repo repair scenarios:** extend deterministic validators to multi-file codebases with unit tests.
2. **Context interventions:** compare model handoff vs context pruning/summarization policies under the same perturbation schedule.
3. **Ablations:** probe branch count, embedding model choice for SCR, and budget/cost trade-offs.

## 7. Conclusion
We propose a reproducible framework for evaluating tool-using LLM agents under non-stationary constraints. The key practical contribution is robust observability when token-level confidence is unavailable: SCR probes and outcome-based validators remain informative across heterogeneous providers. This enables paper-ready benchmark tables that measure both *success* and *stability under shocks*.

## References
References will be finalized from `docs/references.bib` in the camera-ready version.
