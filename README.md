# Entropic Dynamics of Large Language Models Under Non-Stationary Task Constraints

## Abstract
Current evaluations of Agentic Systems focus on Static Solvability. However, real-world deployment involves Non-Stationary Objectives. Anecdotal evidence suggests Agents exhibit "Cognitive Collapse" when forced to modify their own outputs. This research aims to move beyond "success rates" and mathematically quantify Inference-Time Entropic Collapse, correlating the rise in token-level entropy and code structural complexity with the failure to adapt to dynamic requirement injection.

## System Architecture: The "Entropic Stress-Test" Framework v2.0

Incorporating insights from "Agentic Entropy-Balanced Policy Optimization" (AEPO).

### 1. High-Level Logic Flow
1.  **Phase 1: Normal Operation:** Agent executes initial task. Monitor logs baseline entropy.
2.  **Phase 2: The Perturbation:** Orchestrator injects a requirement change.
3.  **Critical Measurement Point (Branching Probe):**
    *   Orchestrator forces Agent to generate Top-5 responses.
    *   Monitor calculates **Semantic Collapse Ratio (SCR)** based on divergence.
4.  **Recovery Cycle:**
    *   Agent continues execution on the "Best" path.
    *   Monitor calculates **Information Gain Efficiency (IGE)** after tool use.
    *   **Hysteresis Intervention:** If entropy > threshold for 3+ consecutive turns ("Persistent Panic"), the Orchestrator resets the context.

### 2. Component Specifications

#### A. The Orchestrator (The Controller)
*   **Mode Switch:** Toggles between `Linear_Run` (temp=0.1) and `Branching_Probe` (temp=0.7, n=5).
*   **Hysteresis Logic:** Only intervenes if "Panic Counter" >= 3.

#### B. The State Monitor (The Calculator)
Now computes Differential Metrics:

*   **Metric 1: Information Gain Efficiency (IGE)**
    *   *Purpose:* Detect "Thrashing" (working without learning).
    *   *Formula:* $IGE = \frac{H_{pre\_tool} - H_{post\_tool}}{\text{Token Cost}}$
    *   *Hypothesis:* Failing agents have IGE $\approx 0$.

*   **Metric 2: Semantic Collapse Ratio (SCR)**
    *   *Purpose:* Measure "Over-branching" (Confusion).
    *   *Formula:* Average pairwise Cosine Distance of Top-5 generated thought embeddings.
    *   *Hypothesis:* High SCR = Total Confusion/Collapse.

*   **Metric 3: The Regressive Debt (RDI)**
    *   *Purpose:* Did it break old tests?

#### C. The Sandbox (Environment)
*   **State Snapshotting:** Supports "Forking" for Branching Probes (measuring intent without executing side-effects 5 times).

## Experiments
1.  **Baseline (Linear):** Standard task execution.
2.  **The Shock (Dynamic):** Injection of conflicting constraints.
3.  **The Rescue (Model Handoff):** Switching models upon entropy spikes.
