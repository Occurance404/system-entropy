# Thesis Framework: Entropic Dynamics of Large Language Models Under Non-Stationary Task Constraints

**Based on:** `paper/CONFERENCE_PAPER.md` and Project Codebase (`src/`)
**Status:** Draft / Framework Design

---

## Abstract
*   **Current Draft:** Use the abstract from `paper/CONFERENCE_PAPER.md`.
*   **Expansion Needed:** Explicitly mention the "Branching Probe" technique and the specific performance gap between standard entropy metrics and the new SCR metric.

---

## Chapter 1: Introduction

### 1.1 Background
*   The rise of Autonomous Agents in Software Engineering (SWE-bench, Devin, etc.).
*   The shift from "Static Solvability" (solving a fixed LeetCode problem) to "Dynamic Maintenance" (long-running tasks).

### 1.2 Problem Statement
*   **"Context Rot" / "Cognitive Collapse":** Define the phenomenon where agents degrade over time, not due to lack of knowledge, but due to confusion from shifting contexts.
*   **The "Silent Killer":** Highlight the specific finding that agents often report high confidence (Low Entropy) while failing (High Semantic Collapse). *Reference `archive/legacy_docs/FINAL_REPORT.md`.*

### 1.3 Research Objectives
1.  To develop a framework for "Stress-Testing" agents with non-stationary objectives (Shocks).
2.  To define and validate the **Semantic Collapse Ratio (SCR)** as a superior metric for agent reliability.
3.  To demonstrate a **"Rescue Protocol"** (Hysteresis-based intervention) that can recover failing agents.

### 1.4 Scope & Limitations
*   **Scope:** Text-based LLM agents, Python/Data Engineering tasks.
*   **Limitations:** Cost of "Branching Probes" (requires 5x inference), focus on single-agent architectures.

---

## Chapter 2: Literature Review

### 2.1 LLM Agents & Cognitive Architectures
*   **ReAct & Chain-of-Thought:** Review Wei et al. (2022) and Yao et al. (2022). Explain how your agent uses a similar loop (`src/agent/real_agent.py`).
*   **Memory & Persistence:** Discuss Generative Agents (Park et al.) and how they handle long contexts.

### 2.2 Evaluation Benchmarks
*   **Static:** SWE-bench, HumanEval (fixed goals).
*   **Dynamic:** Explain why current benchmarks fail to capture "Context Rot" (they don't change requirements mid-stream).

### 2.3 Entropy & Uncertainty Estimation
*   Token-level probability (Perplexity) vs. Semantic Uncertainty (Kuhn et al.).
*   Explain why "RLHF'd models are overconfident."

---

## Chapter 3: Methodology (The Entropic Stress-Test)

*   *Source Code Reference: `src/orchestrator/engine.py`, `src/services/metrics.py`*

### 3.1 System Architecture
*   **The Orchestrator:** The state machine that controls the loop.
*   **The Sandbox (`terminal-bench`):** Explain the Docker-based environment (`src/connectors/tb_connect.py`) providing real shell access.
*   **The Monitor:** The sidecar process observing the agent.

### 3.2 The "Branching Probe" Technique
*   **Definition:** Explain the algorithm where the agent is forked $N$ times at critical steps.
*   **Implementation:** Detail the `generate_multiple` function in `src/agent/real_agent.py`.

### 3.3 Mathematical Definitions of Metrics
*   **Semantic Collapse Ratio (SCR):**
    $$ SCR = \frac{1}{N(N-1)} \sum_{i \neq j} (1 - \cos(\theta_{i,j})) $$
    *Explain how this uses embedding distances to measure "fractured" reasoning.*
*   **Information Gain Efficiency (IGE):**
    *   Measure of "thrashing" (tool use without reducing uncertainty).
*   **Regressive Debt Index (RDI):**
    *   Measure of backward progress (undoing valid work).

### 3.4 The Rescue Protocol
*   **Trigger Logic:** If `SCR > Threshold` or `Panic_Counter > Limit` -> `switch_agent()`.
*   *Reference `experiments/run_rescue_experiment.py` logic.*

---

## Chapter 4: Experimental Design

*   *Source Code Reference: `src/scenarios/definitions.py`*

### 4.1 The "Drug Filter Shock" Scenario
*   **Phase 1:** Initial Goal (Filter for `price < 100`).
*   **The Perturbation (Step 4):** "Manager" changes requirements (`price > 100`).
*   **Phase 2:** Observation of adaptation vs. collapse.

### 4.2 Experimental Controls
*   **Models Tested:** Comparison between "Weak" (primary) and "Strong" (rescue) models.
*   **Baselines:** Standard agent run without branching probes.

---

## Chapter 5: Results & Analysis

### 5.1 The "Silent Killer" Phenomenon
*   **Data:** Present the divergence between Entropy (flat) and SCR (spiking).
*   *Figure:* Use the plot from `analysis/visualize_results.py` (Figure 1 in Conference Paper).

### 5.2 Metric Validation
*   Show correlation tables: Does high SCR predict task failure?
*   Show that standard Entropy *failed* to predict failure.

### 5.3 Efficacy of the Rescue Protocol
*   Did the "Rescue" intervention actually save the task?
*   Cost analysis: How many tokens were saved by intervening early vs. letting the agent fail?

---

## Chapter 6: Discussion

### 6.1 Implications for Agent Ops
*   Passive monitoring (logs) is not enough.
*   Active probing (Branching) is required for safety-critical agents.

### 6.2 The Cost-Reliability Trade-off
*   Discuss the 5x compute cost of SCR. Is it worth it? (Yes, for high-stakes software engineering).

---

## Chapter 7: Conclusion & Future Work

### 7.1 Summary of Contributions
*   The Framework, The Metrics (SCR/IGE), The Findings (Silent Killer).

### 7.2 Future Work
*   **Repo Repair:** Scaling to multi-file repositories.
*   **Context Surgery:** Pruning context instead of swapping models.

---

## References
*   (List from `paper/CONFERENCE_PAPER.md` and others found during literature review)

## Appendices
*   **A: Algorithm Pseudocode:** For the Orchestrator loop.
*   **B: Prompt Templates:** The system prompts used in `src/agent/real_agent.py`.
