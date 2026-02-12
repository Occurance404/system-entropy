# Project State Map (Short, Current View)

This is a lightweight map of what exists, what works, and what is still missing.
It is meant to reduce cognitive load when you come back after a break.

## Core System (Implemented)
- Orchestrator step loop, perturbation injection, and probe control live in `src/orchestrator/core/orchestrator.py`.
- Branching probe execution and SCR calculation are wired through `src/orchestrator/core/probes.py`.
- SCR uses cosine distance over sentence embeddings via `src/services/metrics.py`.
- Agent wrapper for tool actions and probe generations is in `src/agent/real_agent.py`.
- Scenario definitions + setup + validators are in `src/scenarios/definitions.py`, `src/scenarios/setup_ops.py`, and `src/scenarios/validation_ops.py`.
- Deterministic sandbox execution supports Docker or local fallback (configured in the orchestrator).

## Data and Logs (Present)
- Run logs live in `logs/terminal_bench` and other subfolders under `logs/`.
- Run artifacts and manifests are written under `data/run_artifacts`.
- Smoke test is available in `scripts/smoke_test.py` (note: it clears `logs/terminal_bench`).

## Experiments and Analysis (Implemented)
- Benchmark runner: `experiments/run_benchmark.py` (multi-scenario, multi-model, repeats).
- Manual small runs: `experiments/simulate.py` (single scenario/steps).
- Aggregate analysis: `analysis/analyze_benchmark.py`.
- Visualization: `analysis/visualize_results.py` and `analysis/visualize_aggregate.py`.

## Docs and Paper (In Progress)
- Main paper draft: `paper/CONFERENCE_PAPER.md` (methodology updated to match code).
- Thesis materials in `thesis/` and `thesis/chapters/`.
- New scenario ideas exist only in `docs/SCENARIO_DESIGNS.md` (not coded yet).

## Known Gaps / Missing Work
- The new scenario designs are not implemented in `src/scenarios/definitions.py` yet.
- Baseline context policies (windowed/summarized) are not implemented.
- Some plots are still too dense for long runs; `analysis/visualize_results.py` was updated to auto-thin and drop empty panels.

## Current Risks / Gotchas
- Token logprobs are often unavailable; entropy is frequently null and panic stays flat.
- SCR depends on SentenceTransformer embeddings; if the embedding model fails to load, SCR is null.
- Probe calls add extra token cost; track `probe_total_tokens`.

## Next Safe Moves (Low Risk)
- Run small probe-only sanity checks on 1-2 scenarios to verify metrics.
- Implement baseline context policies in a separate, gated branch.
- Convert key plots to a clean two-figure set for the paper (main + appendix).

## Weekly Focus (Checklist)
- Choose 1-2 scenarios to run (write down the IDs).
- Run 3 repeats per scenario with probes on.
- Generate plots and one summary CSV.
- Write 5-7 bullet takeaways for the week.
- Decide the next single improvement or experiment to run.
