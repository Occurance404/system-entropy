# Entropic Dynamics of Large Language Models Under Non-Stationary Task Constraints

**Author:** [Your Name]
**Date:** December 2025
**Status:** Draft V2 (Expanded Introduction & Theory)

---

## Abstract

The deployment of Large Language Models (LLMs) as autonomous agents in software engineering is predicated on their ability to maintain coherent reasoning over long horizons. However, current evaluation frameworks, such as SWE-bench or HumanEval, focus predominantly on "static solvability"—the ability to solve a fixed problem statement. This thesis identifies a critical failure mode in real-world operations: **Non-Stationary Objectives**, where task requirements shift dynamically during execution. We observe that under these conditions, LLM agents exhibit "Context Rot" or "Cognitive Collapse," a state where the agent persists in obsolete plans despite contradictory new instructions. Crucially, we identify a "Silent Killer" phenomenon: Reinforcement Learning from Human Feedback (RLHF) optimization causes models to maintain high confidence (low token entropy) even as their internal logic fractures. To address this, we introduce the **Entropic Stress-Test Framework**, a novel evaluation system that subjects agents to controlled "shocks." We propose a new metric, the **Semantic Collapse Ratio (SCR)**, which utilizes a "Branching Probe" technique to measure the vector-space divergence of parallel reasoning paths. Our experiments demonstrate that SCR successfully detects cognitive collapse (spiking to 0.65) where standard entropy metrics fail (flatlining at 0.0), offering a robust new signal for agent reliability and automated intervention.

---

## Chapter 1: Introduction

### 1.1 The Illusion of Competence

The rapid evolution of Large Language Models (LLMs) has ushered in a new era of "Agentic AI"—systems designed not just to chat, but to execute complex, multi-step workflows autonomously. From Devin to open-source implementations like AutoGPT, the promise is an AI that can act as a junior developer: writing code, running tests, and iterating on errors.

However, this promise masks a dangerous fragility. Modern LLMs are fine-tuned using Reinforcement Learning from Human Feedback (RLHF) to be helpful, harmless, and *confident*. While this makes for a pleasant chatbot, it creates a catastrophic failure mode for an autonomous agent. When an RLHF-optimized model encounters a situation it does not understand—such as a sudden change in project requirements—it does not pause to ask for clarification. Instead, it "hallucinates competence." It continues to generate fluent, syntactically correct code that is semantically totally divorced from the new reality.

We term this phenomenon **"The Illusion of Competence."** Standard observability metrics, specifically Token-Level Entropy (Perplexity), fail to pierce this veil. A model can be 99% confident (Entropy $\approx$ 0) that the next word is "function," even if that function implements a feature that was deleted five minutes ago. For safety-critical applications in banking, healthcare, or infrastructure, this "silent failure" is unacceptable.

### 1.2 The Problem: Context Rot and Non-Stationary Tasks

Real-world software engineering is rarely static. A developer does not simply "solve a LeetCode problem." They work in a dynamic environment where a manager might change a database schema halfway through the day, or a library update might deprecate a key function.

Current benchmarks fail to capture this dynamism. They measure **Static Solvability**: given a frozen snapshot of a repo and a fixed issue description, can the agent fix it?
This thesis addresses **Dynamic Maintenance**: can the agent survive a "Shock"—a sudden, contradictory update to its instructions?

We observe that when subjected to such shocks, agents suffer from **"Context Rot."** The large context window, usually an asset, becomes a liability. The agent's short-term memory is filled with the "old" plan. When the "new" instruction arrives, it fights against the inertia of the generated tokens. The agent acknowledges the change ("Understood, I will update the schema") but then immediately hallucinates the old schema in the very next code block. This is **Cognitive Collapse**—the dissociation of intent from action.

### 1.3 Research Objectives

This thesis aims to mathematically quantify and detect this collapse before it causes damage. Our primary objectives are:

1.  **To Define and Quantify "Cognitive Collapse":** We move beyond binary success/fail metrics to define a continuous measure of "internal confusion." We hypothesize that a confused agent, if forced to think twice, will have *divergent* thoughts.
2.  **To Develop the Entropic Stress-Test Framework:** We build a rigorous, reproducible experimental testbed (`terminal-bench` + `orchestrator`) that subjects agents to controlled "Entropic Shocks" (conflicting requirements) in a sandboxed environment.
3.  **To Validate the Semantic Collapse Ratio (SCR):** We introduce a novel metric, SCR, derived from the cosine distance of "Branching Probes." We aim to prove that SCR is a superior predictor of task failure than standard Entropy, specifically in the context of RLHF-optimized models.

### 1.4 Thesis Structure

*   **Chapter 2** critiques the limitations of Shannon Entropy in the era of RLHF and derives the theoretical basis for Vector Space Divergence.
*   **Chapter 3** details the Methodology, describing the architecture of the Orchestrator, the Dockerized Sandbox, and the mathematical formulation of the SCR and IGE metrics.
*   **Chapter 4** presents the "Drug Filter Shock" and "Scale" experiments, providing empirical evidence of the "Silent Killer" phenomenon.
*   **Chapter 5** discusses the implications for "Agent Ops" and the future of self-healing AI systems.

---

## Chapter 2: The Theory of Entropic Collapse

### 2.1 Why Entropy is Broken
*   **Definition:** Shannon Entropy measures the uncertainty of the *next token*.
    $$ H(x) = - \sum p(x) \log p(x) $$
*   **The Flaw:** RLHF (Reinforcement Learning from Human Feedback) forces models to collapse their probability distribution to a single "preferred" answer.
*   **Observation:** In our experiments, failing agents often report $H(x) \approx 0.0$. They are "Confident but Wrong."

### 2.2 The Solution: Vector Space Divergence
*   **Philosophy:** If a model is truly confident, it should have only *one* plan. If it is confused, it might have *many* potential plans, even if it only outputs one.
*   **The "Branching Probe":** We force the model to reveal its hidden confusion by asking it to generate $N=5$ parallel "next steps" from the exact same state.
*   **Semantic Collapse Ratio (SCR):** We measure the cosine distance between these 5 parallel thoughts.
    *   **Low SCR:** All thoughts are identical (True Confidence).
    *   **High SCR:** Thoughts diverge (Hidden Confusion).

### 1.5 The Economics of Reliability: Why Not Just Use GPT-4?

A common critique of multi-agent "rescue" architectures is the "Ferrari Fallacy": *If a superior model exists (e.g., GPT-4o, Claude 3.5 Sonnet), why risk failure with a weaker model (e.g., Llama-3-70B) in the first place?*

This thesis argues that **Dynamic Compute Allocation** is not merely a cost-saving measure, but a reliability necessity.
1.  **Cost & Latency:** Running State-of-the-Art (SOTA) models for every step of a long-horizon task is economically prohibitive and latently inefficient. 90% of software engineering consists of rote tasks (boilerplate, syntax) that do not require SOTA reasoning.
2.  **Context Saturation:** Even SOTA models suffer from Context Rot. Simply upgrading the model does not solve the problem of an incoherent context window.
3.  **The "Fresh Eyes" Effect:** The "Rescue" protocol defined in this thesis is not just about intelligence; it is about **discontinuity**. By swapping agents when SCR spikes, we force a break in the "chain of confusion," often initializing the new agent with a summarized or pruned context. This provides a clean slate that a single, continuous agent session cannot achieve.

Thus, the **Semantic Collapse Ratio (SCR)** serves as the "Switching Signal" for an optimized cognitive pipeline: run cheap/fast until confusion is detected, then surgically apply expensive/slow reasoning to resolve the blockage.

---

## Chapter 3: Methodology

### 3.1 System Architecture
*   **The Orchestrator (`src/orchestrator/engine.py`):** A state machine that manages the simulation. It freezes the agent at every step to run the "Branching Probe."
*   **The Sandbox (`terminal-bench`):** A Dockerized environment ensuring the agent has real consequences (creating files, running code). We do not use simulated mocks; if the agent deletes a file, it is gone.

### 3.2 The Metrics Suite

We introduce three novel metrics to quantify agentic resilience.

**Metric 1: Semantic Collapse Ratio (SCR)**
SCR quantifies the divergence of the agent's latent reasoning space. It serves as a proxy for "internal confusion."
$$ SCR = \frac{1}{N(N-1)} \sum_{i \neq j} (1 - \cos(\mathbf{e}_i, \mathbf{e}_j)) $$
Where:
*   $\mathbf{e}_i$ is the vector embedding of the $i$-th generated reasoning branch.
*   $N=5$ is the number of parallel branches generated during the probe.
*   $\cos(\mathbf{a}, \mathbf{b})$ is the cosine similarity function.
A value of 0.0 indicates perfect alignment (dogmatic confidence). A value approaching 1.0 indicates total semantic dissociation (panic).

**Metric 2: Information Gain Efficiency (IGE)**
IGE measures the utility of an action relative to its computational cost. It detects "thrashing"—where an agent burns tokens without reducing uncertainty.
$$ IGE = \frac{H_{pre} - H_{post}}{C_{tokens}} $$
Where:
*   $H_{pre}$ is the Shannon Entropy of the agent's action distribution *before* tool execution.
*   $H_{post}$ is the Shannon Entropy *after* observing the tool output.
*   $C_{tokens}$ is the number of tokens consumed by the action.

**Metric 3: Regressive Debt Index (RDI)**
RDI measures how far the agent's current intent has drifted from the "Ground Truth" optimal path.
$$ RDI = 1 - \cos(\mathbf{e}_{current}, \mathbf{e}_{truth}) $$
Where:
*   $\mathbf{e}_{current}$ is the embedding of the agent's generated plan.
*   $\mathbf{e}_{truth}$ is the embedding of the known golden path step.
High RDI indicates the agent is actively working on the wrong goal.

### 3.3 Experimental Design: "The Shock"
*   We define **Non-Stationary Tasks** where the Goal Function $G(t)$ changes over time.
    *   *Phase 1:* $G_1$ (e.g., "Filter by Weight").
    *   *Phase 2:* The Shock (Step $T_{shock}$).
    *   *Phase 3:* $G_2$ (e.g., "Filter by Molecular Mass").
*   **Hypothesis:** A robust agent adapts. A rotting agent persists with $G_1$ while hallucinating $G_2$.

### 3.4 Critical Design Decisions

Our framework diverges from standard evaluation methodologies (e.g., SWE-bench) on three fundamental principles.

**1. Rejection of Self-Reported Confidence**
We explicitly reject "Self-Correction" or "Reflexion" loops as a primary metric for reliability. RLHF training biases models toward agreeableness and perceived competence. When asked, "Are you sure?", a collapsing agent will typically hallucinate a justification for its error rather than admit uncertainty. Therefore, our **Branching Probe** is designed to be *involuntary*. By sampling the latent space directly, we bypass the model's "PR department" and measure its actual cognitive stability.

**2. Semantic vs. Syntactic Drift**
Standard consistency metrics (like BLEU or ROUGE) measure lexical overlap. In high-level reasoning tasks, these are insufficient. An agent might output "I will delete the file" in one branch and "I am removing the document" in another. Syntactically, these are divergent; semantically, they are identical. Our **SCR metric** utilizes vector embeddings to ignore superficial wording differences and detect only true fractures in *intent*.

**3. Execution as the Sole Arbiter of Truth**
Text-based benchmarks allow for the "Illusion of Competence"—code that *looks* correct but fails to run. Cognitive collapse often manifests not as gibberish, but as plausible-looking code that references non-existent variables from a previous task state. By enforcing a **Dockerized Sandbox**, we ensure that "Context Rot" has immediate, measurable consequences (e.g., `FileNotFoundError`), grounding our metrics in operational reality rather than linguistic fluency.

### 3.5 The Branching Probe Algorithm

To operationalize the Semantic Collapse Ratio, we implement a rigorous "Branching Probe" algorithm. This process occurs at every step $t$ of the agent's trajectory, prior to the execution of any tool.

1.  **State Freeze:** The Orchestrator pauses the simulation. The agent's context window $C_t$ (including all history, tool outputs, and system prompts) is preserved.
2.  **Parallel Generation:** We fork the execution into $N=5$ parallel threads. In each thread, the agent is prompted with the exact same context $C_t$. We employ a slightly elevated temperature ($T=0.7$) to allow for latent probability distribution sampling, ensuring we capture the full "cone of possibility" the agent is considering.
3.  **Intent Extraction:** For each branch $b_i$, we extract the "Action Intent"—the specific tool call or reasoning block generated.
4.  **Vector Embedding:** Each $b_i$ is passed to an external embedding model (specifically `all-MiniLM-L6-v2`), mapping the textual intent to a dense vector representation $\mathbf{e}_i \in \mathbb{R}^{384}$.
5.  **Divergence Calculation:** We compute the pairwise cosine distances between all vectors to populate the SCR formula defined in Section 3.2.
6.  **Collapse:** The branches are discarded. The Orchestrator selects a single action (via the primary temperature setting) to proceed to state $t+1$, ensuring the probe itself does not contaminate the agent's memory.

### 3.6 Service-Oriented Architecture

To ensure the integrity of the measurements, the system employs a strict separation of concerns:

*   **The Agent (Subject):** Operates in a stateless manner per request, unaware it is being probed.
*   **The Orchestrator (Controller):** Manages the experiment lifecycle. It is responsible for injecting "Shocks" (System Messages) based on the Scenario definition.
*   **The Metric Service (Observer):** An isolated microservice (or dedicated module) responsible for calculating Entropy, embeddings, and SCR. This isolation prevents the heavy computational load of embedding calculation from affecting the latency or state of the agent's interaction loop.
*   **The Sandbox (Environment):** A Docker container (using `terminal-bench`) that provides a persistent filesystem. This persistence is crucial: if an agent creates a file in Step 1, it *must* be there in Step 10. If the agent "collapses" and forgets where the file is, the Sandbox provides the ground truth error (`FileNotFoundError`), which serves as the "consequence" that drives the experiment forward.

---

## Chapter 4: Experimental Setup

To validate the "Silent Killer" hypothesis, we designed three distinct experimental scenarios. Each scenario is constructed to establish a "Golden Path" (success) and then disrupt it with a specific "Entropic Shock" (perturbation).

### 4.1 Scenario A: The Baseline (Drug Filter Shock)
*   **Domain:** Data Science / Pandas.
*   **The Task:** The agent is given a CSV file of pharmaceutical data and asked to filter it based on specific criteria (Weight, Solubility).
*   **The Golden Path:** The agent writes a Python script using `pandas` to filter the dataframe and saves the result. This establishes a baseline for "Linear Solvability."
*   **The Shock (Step 4):** "Constraint Reduction." The agent is told that the 'Weight' column is unreliable and must now use an external API to fetch molecular mass.
*   **Why this matters:** This tests **Plasticity**. Can the agent modify valid code it just wrote, or will it refuse to break its own working artifact?

### 4.2 Scenario B: The Stress Test (Distributed System Implementation)
*   **Domain:** Systems Programming / Algorithm Design.
*   **The Task:** "Implement a thread-safe, persistent Key-Value store with ACID transaction support (BEGIN, COMMIT, ROLLBACK)."
*   **The Golden Path:** A complex, multi-file Python project involving file locking, serialization, and class inheritance. This requires long-horizon reasoning and strict architectural consistency.
*   **The Condition:** **No Rescue.** Unlike Scenario A, no interventions are provided. The agent must self-correct.
*   **The Shock (Implicit):** The sheer complexity of the task serves as an "Internal Shock." As the codebase grows, the context window fills. We hypothesize that "Context Rot" will occur naturally as the agent forgets its own variable names from Step 1.
*   **Why this matters:** This tests **Endurance**. It simulates the "Senior Developer" workload where guidance is minimal and architectural coherence is paramount.

### 4.3 Scenario C: The Cognitive Load Test (Financial Fraud Detection)
*   **Domain:** Forensic Analysis / Multi-Modal Correlation.
*   **The Task:** "Analyze `transactions.csv` and `kyc_data.json`. Identify a circular trading ring of at least 4 accounts. Calculate total laundered volume."
*   **The Golden Path:** The agent must cross-reference two distinct datasets (Structured vs. Semi-Structured). It must maintain a "mental model" of the suspect graph.
*   **The Condition:** **No Rescue.**
*   **The Shock (Implicit):** Information Overload. The agent is flooded with irrelevant data.
*   **Why this matters:** This tests **Focus**. Can the agent distinguish signal from noise, or will the SCR metric reveal that it is "hallucinating" patterns that don't exist?

### 4.1 The "Probe" Implementation
*   *Code Reference:* `src/agent/real_agent.py` -> `generate_multiple()`
*   We use `asyncio` to fire 5 parallel requests to the LLM API (OpenAI/vLLM).
*   We use `all-MiniLM-L6-v2` (via `sentence-transformers`) to embed these 5 responses into vector space.

### 4.2 The "Shock" Injection
*   *Code Reference:* `src/scenarios/definitions.py`
*   Shocks are defined as "Perturbations" in the scenario config. They are injected as System Messages that override previous instructions.

---

## Chapter 5: Preliminary Results (The "Silent Killer")

### 5.1 The Golden Run (Drug Filter Shock)
*   **[PLACEHOLDER FOR IMAGE: data/results/experiment_summary.png]**
*   **Analysis:**
    *   At Step 1-3 (Linear Task): Entropy is Low, SCR is Low.
    *   **At Step 4 (The Shock):**
        *   **Entropy:** Remains **Flat (0.0)**. The model does *not* report uncertainty.
        *   **SCR:** Spikes to **0.24**. The Vector Space reveals the confusion.
    *   *Conclusion:* This confirms the "Silent Killer" hypothesis. The model lied, but the embeddings told the truth.

### 5.2 The Scale Experiment (Scale & Limitations)
*   **[PLACEHOLDER FOR IMAGE: data/results/scale_scr_trajectory.png]**
*   **Observation:** In a scale run of N=50, we observed instances where SCR spiked to **0.65** under shock.
*   **Limitations:** High-frequency probing triggers API Rate Limits (429 Errors) on public models. This necessitates a "Slow & Steady" approach for future data collection.

---

## Chapter 6: Discussion & Future Work

### 6.1 The Cost of Reliability
*   **Trade-off:** SCR is expensive (5x compute).
*   **Argument:** For safety-critical code (e.g., banking, healthcare), paying 5x compute to detect "Silent Rot" is cheaper than a production outage.

### 6.2 The "Rescue" Protocol
*   **Concept:** We don't just want to detect rot; we want to fix it.
*   **Mechanism:** When `SCR > Threshold`, we pause the "Junior" agent (e.g., GPT-3.5) and hot-swap in a "Senior" agent (e.g., GPT-4o) to refactor the plan.

---

## Chapter 7: Conclusion
We have mathematically quantified "Cognitive Collapse." By moving beyond static benchmarks to **Entropic Stress-Testing**, we can build autonomous agents that know when they are confused—even if they are trained to sound confident.

---

## References
1.  Wei, J., et al. (2022). Chain-of-Thought Prompting.
2.  Park, J. S., et al. (2023). Generative Agents.
3.  [Your Codebase References]
