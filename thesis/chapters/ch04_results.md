CHAPTER 4 — EXPERIMENTAL RESULTS

This chapter summarizes the latest runs (Dec 09, 2025) of the Entropic Stress-Test Framework. All runs used `deepseek/deepseek-v3.2` with 20-step limits for shock scenarios and 30 steps for the legacy refactor challenge. Metrics were logged each step; no rescue handoff was enabled in these runs (Zhou2024Consistency; Chen2023ToolEmu).

4.1 Experimental Setup (Current)

- Sandbox: fresh container per run; real tool execution.
- Probes: Branching probes on shock steps; entropy recorded every step.
- Runs covered three shock scenarios (Drug Filter, File Organizer, Data Pipeline) and a control-style Legacy Refactor Challenge.

4.2 Latest Run Snapshot (Dec 2025)

| Scenario (file) | Steps | Events (llm/tool/perturb) | Max SCR | Max Entropy | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Drug Filter Shock (sim_baseline_drug_filter_shock_20251209_153021.jsonl) | 20 | 12 / 6 / 2 | 0.237 | 0.160 | Shock applied twice; SCR spike mild; entropy low |
| Drug Filter Shock (sim_baseline_drug_filter_shock_20251209_133158.jsonl) | 20 | 3 / 15 / 2 | 0.230 | 0.231 | Tool-heavy loop after shock; entropy stays low/moderate |
| File Organizer Shock (sim_baseline_file_organizer_shock_20251209_153606.jsonl) | 20 | 14 / 5 / 1 | **0.337** | 0.076 | Highest SCR in this batch; entropy flat |
| File Organizer Shock (sim_baseline_file_organizer_shock_20251209_133630.jsonl) | 20 | 3 / 16 / 1 | 0.068 | 0.142 | Minimal SCR change despite shock; tool spam |
| Data Pipeline Shock (sim_baseline_data_pipeline_shock_20251209_154156.jsonl) | 20 | 16 / 3 / 1 | 0.270 | 0.050 | SCR rises on shock; entropy near zero |
| Data Pipeline Shock (sim_baseline_data_pipeline_shock_20251209_134210.jsonl) | 20 | 12 / 7 / 1 | 0.025 | **0.267** | No SCR probes fired; entropy highest of the batch |
| Legacy Refactor Challenge (sim_baseline_legacy_refactor_challenge_20251209_151022.jsonl) | 30 | 19 / 11 / 0 | 0.000 | 0.127 | Control task; no probes; long run with low entropy |
| Legacy Refactor Challenge (sim_baseline_legacy_refactor_challenge_20251209_145552.jsonl) | 30 | 1 / 29 / 0 | 0.000 | 0.138 | Tool spam; no SCR |
| Legacy Refactor Challenge (sim_baseline_legacy_refactor_challenge_20251209_144921.jsonl) | 30 | 1 / 29 / 0 | 0.000 | **0.239** | Highest entropy among control runs; no SCR |

4.3 Patterns Observed

- SCR spikes align with shocks: File Organizer (0.337) and Data Pipeline (0.270) show clear divergence at the perturbation point; Drug Filter shows smaller rises (~0.23). Runs without probes report SCR≈0 (Zhou2024Consistency).
- Entropy stays low even when behavior degrades (0.05–0.23 range); higher entropy (0.267) still failed to warn (Chen2025Overconfidence; Perez2024Mislead).
- Stagnation loops persist: tool_execution dominates several runs (e.g., 16–29 tool calls with minimal llm_reply), mirroring prior “ls/loop” failures even when SCR is low or absent (Chen2023ToolEmu).
- Panic/rescue not triggered: panic_counter stayed at 0 in these logs, so no escalations occurred despite loops.

4.4 Scenario Notes

- Drug Filter Shock: Two shocks injected; modest SCR rise (~0.23) and low entropy; behavior still drifted into tool repetition after acknowledging the change.
- File Organizer Shock: One run showed strong divergence (0.337) at the shock; another barely moved (0.068) but still spammed tools—suggesting shock detection depends on probe coverage.
- Data Pipeline Shock: One run produced SCR 0.270 with near-zero entropy; another skipped probes (SCR 0.025) yet showed the highest entropy of the batch, highlighting probe timing sensitivity.
- Legacy Refactor Challenge: No perturbations and no SCR; entropy stayed low-to-moderate (≤0.239) while tool spam dominated, reinforcing that stagnation occurs even without shocks (Jimenez2024SWEBenchPlus motivates why static tasks can still mask instability).

4.5 Artifacts to Reference

- Plots: `data/results/summary_sim_baseline_drug_filter_shock_20251209_153021.png`, `data/results/summary_sim_baseline_file_organizer_shock_20251209_153606.png`, `data/results/summary_sim_baseline_data_pipeline_shock_20251209_154156.png`, `data/results/summary_sim_baseline_legacy_refactor_challenge_20251209_151022.png`, plus aggregate views `data/results/experiment_summary.png`, `data/results/entropy_comparison.png`, `data/results/mini_swe_agent_metrics.png`.
- Logs: `logs/rescue/*.jsonl` for each run listed above.

4.6 Chapter Summary

The refreshed runs confirm the central finding: token entropy stays calm while internal coherence fractures under shocks. SCR provides the only meaningful early signal when probes fire, but probe timing and coverage matter—runs without probes report SCR≈0 even when behavior stalls. Stagnation loops persist in both shock and control settings, underscoring the need for rescue/interrupt logic beyond entropy monitoring.
