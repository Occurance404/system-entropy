# Baseline Comparison Plan (No Code Yet)

Goal: compare context policies while holding everything else fixed. This gives a publishable, controlled result without changing model or scenarios.

## Policies to Compare
1) **Append-only (current)**  
   Keep full history; no summarization.

2) **Sliding window**  
   Keep system prompt + initial goal + last *K* steps only. Older steps are dropped.

3) **Summarize-then-window**  
   When history exceeds a threshold, summarize the old segment into a short memory block, then keep last *K* steps.

## Controlled Variables (Must Stay Fixed)
- Same model and temperature.
- Same scenarios and perturbation schedules.
- Same max steps and validation interval.
- Same probe mode and branch count.
- Same embedding model for SCR.

## Metrics to Report
Primary:
- Success rate (validator pass/fail).
- Peak SCR at shocks.
- Token usage (including probe calls).

Secondary:
- Entropy coverage.
- Median steps to completion.

## Implementation Plan (Later)
1) Add a `context_policy` flag in the orchestrator or agent wrapper.
2) Apply the policy right before `agent.get_next_action(...)`.
3) Log the policy name in the run manifest.
4) Track extra summary tokens separately (e.g., `summary_total_tokens`).

## Suggested Defaults
- Window size: last 8 steps.
- Summary trigger: when history exceeds ~8k characters or a fixed step count.
- Summary length: 6-10 bullet lines, single paragraph max.
- Probe mode: `shock` (default) to control cost.

## Experimental Matrix (Minimal Publishable)
- Models: 1 model (local).
- Scenarios: 5-7 (existing hard-mode set).
- Repeats: 3 per (policy, scenario).

Total runs = policies * scenarios * repeats.
Start with 2 policies (append-only vs sliding window) if time is tight.

## Expected Outcomes (What We Want to Test)
- Sliding window reduces context drift but may increase failure if critical context is lost.
- Summarize-then-window retains intent and reduces SCR spikes after shocks.
- Success and SCR may decouple; SCR can detect instability even when success stays high.

## Decision Gate
- If policy B or C reduces SCR without harming success rate, it becomes the baseline improvement.
- If SCR does not change across policies, focus the paper on SCR as a diagnostic metric instead of a solution.
