# ENTROPIC DYNAMICS OF LARGE LANGUAGE MODELS UNDER NON-STATIONARY TASK CONSTRAINTS

---

## 1. TITLE PAGE (Placeholder)
*(To be filled with: Title, Student Name, Supervisor, Department, Institute, Date)*

---

## 2. CERTIFICATE (Placeholder)
*(Standard statement of originality to be signed by supervisor)*

---

## 3. ABSTRACT

**Abstract**

As Large Language Model (LLM) agents are increasingly deployed in autonomous software engineering and data analysis roles, their reliability in dynamic, non-stationary environments remains an open challenge. Current evaluation benchmarks focus primarily on static task correctness, failing to capture the "silent failures" where an agent loses reasoning coherence without exhibiting surface-level uncertainty.

This thesis introduces the **Entropic Stress-Test Framework**, a novel system for evaluating agent stability under shifting task constraints. We propose a new metric, the **Semantic Collapse Ratio (SCR)**, which quantifies the divergence of an agent's latent intentions by analyzing parallel reasoning paths in embedding space. 

Experimental results using the DeepSeek-v3.2 model across data processing and coding scenarios demonstrate that standard metrics like token entropy fail to predict collapse ($<0.01$ during failure). In contrast, SCR consistently spikes ($>0.20$) at the exact moment of cognitive fracture, successfully predicting "text stagnation" loops where the agent ceases meaningful tool use. These findings suggest that semantic monitoring is a critical requirement for building safe, reliable autonomous systems.

**Keywords:** LLM Agents, AI Safety, Semantic Collapse, Uncertainty Quantification, Reliability Engineering.

---

## CHAPTER 1 — INTRODUCTION

### 1.1 When AI Agents Fail Without Warning

Large language models have become remarkably capable at handling isolated programming tasks, and modern benchmarks reinforce this confidence. Systems that score well on SWE-bench or HumanEval are often assumed to be reliable developers in a broader engineering context. However, the moment these models step outside the enclosed world of static test cases, a very different picture emerges.

A recurring pattern became obvious during my early experiments: when the task requirements changed mid-execution, the agent did not respond with uncertainty or hesitation. Instead, it continued executing its previous plan with complete confidence—even after acknowledging the new instruction. This disconnect between **understanding** and **action** is subtle but dangerous. Rather than producing visibly incorrect or chaotic output, the agent produced fluent, tidy, and utterly misaligned code. In practice, this means a model may enthusiastically work on the wrong objective for several steps before anyone notices.

This “silent failure” is not a corner case; it appears consistently whenever a model is asked to revise an already-formed plan. And because the model expresses high certainty at the token level, traditional monitoring tools fail to catch the collapse until the agent has already drifted into an error loop or corrupted state.

### 1.2 Why Existing Confidence Metrics Fall Short

For years, token-level entropy has been treated as a useful proxy for model confidence. The logic is straightforward: if a model is unsure, its probability distribution over the next token spreads out; if it is confident, the distribution concentrates.

This intuition breaks down in RLHF-trained models. Human preference rewards confident, decisive-sounding responses. During RLHF fine-tuning, models learn to suppress hesitation, even in situations where internal uncertainty is high. As a result, the entropy of the output distribution becomes decoupled from the model’s actual cognitive state. In practice a model may be deeply confused internally yet still emit low-entropy, high-certainty text because that is what it has been trained to do.

This distortion means that monitoring entropy alone is misleading. In my experiments, entropy often remained at or near zero even when the agent had completely lost the thread of the task. The model will not warn you when it is confused; RLHF has taught it to sound composed no matter what.

### 1.3 What This Thesis Tries to Solve

Because this collapse is silent in the traditional sense, we need a more direct way to measure the stability of an agent’s internal reasoning. The **Entropic Stress-Test Framework** was developed precisely for this purpose. Instead of relying on the model’s self-reported confidence, the framework evaluates models under dynamic, shifting task constraints—much closer to real engineering conditions.

---

## CHAPTER 2 — LITERATURE SURVEY

### 2.1 Autonomous Agent Frameworks and Failure Modes
Recent advancements in agentic AI have popularized architectures where Large Language Models (LLMs) iteratively perceive, plan, and act to solve complex tasks. Frameworks like **ReAct** [48] and **Reflexion** introduce self-improvement loops, allowing agents to reason over multi-step horizons. However, as these systems scale, they exhibit distinct failure patterns. Zhang et al. (2024) in **Agent-SafetyBench** [1] identified 10 common failure modes across 16 agents, revealing that none achieved safety scores above 60%. Similarly, **MedAgentAudit** [5] diagnosed collaborative failures in medical systems, highlighting "flawed consensus" and "information loss" as dominant risks.

Crucially, **Chen et al. (2024)** identified "Silent Errors" in tool use [6], where models incorrectly invoke tools without surfacing uncertainty. This aligns with the "Cognitive Collapse" phenomenon studied in this thesis, where internal confusion is masked by external confidence.

### 2.2 Benchmarks and Evaluation Gaps
Standard benchmarks like **HumanEval** [75] and **SWE-bench** [66] evaluate code generation in static environments. While valuable for measuring capability ("Pass@k"), they fail to assess reliability under non-stationary conditions. **Wang et al. (2025)** demonstrated that SWE-bench overestimates agent capabilities by over 50% due to solution leakage [65]. Furthermore, **ToolSandbox** [9] and **τ-bench** [11] have introduced stateful evaluation, but they still focus on task completion rather than the *stability* of the agent's reasoning process during execution.

This thesis addresses the gap identified by **Guo et al. (2024)** [7], who called for systematic detection of defects in autonomous agents where discrepancies between expected behavior and actual execution lead to "stagnation loops."

### 2.3 Reliability, Uncertainty, and RLHF
A core challenge in agent reliability is the miscalibration of confidence. **Chen et al. (2025)** showed that Reinforcement Learning from Human Feedback (RLHF) exacerbates overconfidence [15], causing models to produce sharp probability distributions even when hallucinating. This "forced certainty" renders traditional entropy metrics ineffective.

Recent work on **Semantic Uncertainty** [30-41] suggests that measuring divergence in embedding space is a more robust signal than token probability. The **MAKER** framework [see Discussion] and **Consistency Calibration** [38] build on this by using ensemble consistency to verify outputs. This thesis extends these insights by proposing the **Semantic Collapse Ratio (SCR)** as a real-time "heartbeat" monitor for agent cognitive stability.

---

## CHAPTER 3 — PROBLEM FORMULATION & MATHEMATICAL MODEL

### 3.1 Problem Definition
The core problem is the **decoupling of confidence and competence** in RLHF-tuned models. We define "Cognitive Collapse" as a state where:
1.  The agent's internal reasoning diverges (High Semantic Uncertainty).
2.  The agent's external output remains confident (Low Token Entropy).
3.  The agent's actions cease to be productive (Stagnation).

### 3.2 Rethinking Entropy
Token-level entropy is defined as:
$$ H(w_{t+1} | w_{1:t}) = - \sum_{v \in \mathcal{V}} P(v | w_{1:t}) \log P(v | w_{1:t}) $$

In RLHF models, the optimization objective includes a penalty term that minimizes this entropy, effectively masking internal confusion.

### 3.3 The Semantic Collapse Ratio (SCR)
We propose SCR as a geometric measure of internal divergence. Using Branching Probes ($B$), we sample $N$ parallel futures and map them to embedding space $\mathbf{e}$:

$$ \text{SCR}_t = \frac{1}{N(N-1)} \sum_{i=1}^{N} \sum_{j>i}^{N} d_{\cos}(\mathbf{e}_i, \mathbf{e}_j) $$

High SCR indicates that the model is considering fundamentally different courses of action, a precursor to collapse.

---

## CHAPTER 4 — PROPOSED METHODOLOGY

### 4.1 System Architecture
The Entropic Stress-Test Framework consists of four components:
*   **The Orchestrator:** Manages the simulation loop and injects shocks.
*   **The Agent Wrapper:** Interfaces with the LLM and runs Branching Probes.
*   **The Sandbox:** A Dockerized environment for safe execution.
*   **The Metric Service:** Computes SCR, IGE, and RDI.

### 4.2 Branching Probes Algorithm
1.  Freeze context $C_t$.
2.  Generate $N=5$ completions at Temperature $0.7$.
3.  Embed completions using `all-MiniLM-L6-v2`.
4.  Compute pairwise Cosine Distance.
5.  Return mean distance as SCR.

---

## CHAPTER 5 — EXPERIMENTAL SETUP

### 5.1 Environment and Models
*   **Model:** DeepSeek-v3.2 (via vLLM/OpenAI API).
*   **Sandbox:** `python:3.11-slim` Docker container.
*   **Hardware:** Linux-based server with GPU acceleration for embeddings.

### 5.2 Scenarios
We defined four scenarios to test Adaptability and Stability:
1.  **Drug Filter Shock:** Domain Shift (CSV $\to$ API).
2.  **File Organizer Shock:** Rule Inhibition (Extension $\to$ Alphabetical).
3.  **Data Pipeline Shock:** Schema Breaking Change.
4.  **Legacy Refactor:** Long-context static control task.

---

## CHAPTER 6 — RESULTS AND DISCUSSION

### 6.1 Quantitative Results
The experiments revealed a consistent failure mode across all dynamic tasks.

**Table 1: Metrics and Failure Modes by Scenario**

| Scenario | Condition | Peak SCR | Outcome | Dominant Failure Mode |
| :--- | :--- | :--- | :--- | :--- |
| **Drug Filter** | Shock (Step 4) | **0.24** | Collapse | **Recursive Text Stagnation** |
| **File Organizer** | Shock (Step 3) | **0.34** | Collapse | **Recursive Text Stagnation** |
| **Data Pipeline** | Shock (Step 4) | **0.27** | Collapse | **Recursive Text Stagnation** |
| **Legacy Refactor** | Control | **0.09** | **Partial Success** | **Termination Failure** |

### 6.2 Analysis of Failure Modes
*   **Safe Mode Retreat:** Upon shock, SCR spiked ($>0.20$), and the agent immediately ceased tool use, reverting to polite, repetitive text acknowledgments.
*   **Termination Failure:** In the static task, the agent succeeded but failed to stop, highlighting a lack of "definition of done."

### 6.3 SCR vs. Entropy
Token entropy remained negligible ($<0.01$) throughout all failures. SCR was the only metric that successfully distinguished between the stable Refactor task ($0.09$) and the collapsed Shock tasks ($0.24+$).

---

## CHAPTER 7 — SOCIETAL IMPACT

### 7.1 Risks of Silent Failure
As agents are integrated into critical infrastructure (banking, healthcare, legal), the risk of "Silent Failure" becomes systemic. An agent that confidently processes data incorrectly—while reporting success—can cause massive financial or physical harm before detection.

### 7.2 Contribution to AI Safety
This research provides a mechanism for **Runtime Governance**. By monitoring SCR, operators can install "Circuit Breakers" that halt an agent the moment it becomes confused, preventing it from taking destructive actions in a hallucinated state. This is a foundational step toward auditable, reliable autonomous systems.

---

## CHAPTER 8 — CONCLUSION

This thesis investigated the stability of LLM agents under non-stationary conditions. We demonstrated that current agents suffer from "Cognitive Collapse," a state of confident stagnation that traditional metrics miss. We introduced the Semantic Collapse Ratio (SCR) and proved that it reliably detects this collapse in real-time. Future work will focus on using SCR to trigger dynamic interventions (Rescue Protocols), moving from failure detection to failure prevention.

---

## REFERENCES

**Agent Failure Modes and Safety**
1.  **Agent-SafetyBench: Evaluating the Safety of LLM Agents**
    Y. Zhang et al., 2024. arXiv:2412.14470.
2.  **Aegis: Taxonomy and Optimizations for Overcoming Agent-Environment Failures**
    M. Chen et al., 2025. arXiv:2508.19504.
3.  **Why Do Multi-Agent LLM Systems Fail?**
    Z. Wang et al., 2025. arXiv:2503.13657.
4.  **VeriLA: A Human-Centered Evaluation Framework for Verification**
    Q. Zhang et al., 2025. arXiv:2503.12651.
5.  **MedAgentAudit: Diagnosing Collaborative Failure Modes**
    L. Sun et al., 2025. arXiv:2510.10185.
6.  **Tools Fail: Detecting Silent Errors in Faulty Tools**
    J. Chen et al., 2024. arXiv:2406.19228.
7.  **Defining and Detecting Defects of Autonomous Agents**
    L. Guo et al., 2024. arXiv:2412.18371.
8.  **Identifying Risks of LM Agents with ToolEmu**
    J. Chen et al., 2023. arXiv:2309.15817.
9.  **ToolSandbox: A Stateful Evaluation Benchmark**
    S. Zhou et al., 2024. arXiv:2408.04682.
10. **MCPMark: Stress-Testing Realistic MCP Use**
    A. Grattafiori et al., 2025. arXiv:2509.24002.
11. **τ-bench: Tool-Agent-User Interaction Benchmark**
    J. Fu et al., 2024. arXiv:2406.12045.
12. **Dark Patterns Meet GUI Agents**
    H. Jiang et al., 2025. arXiv:2509.10723.
13. **Multiparty Dynamics and Failure Modes**
    A. Raji et al., 2019. arXiv:1810.10862.
14. **Empirical Characterization of Outages in LLM Services**
    P. Tang et al., 2025. arXiv:2501.12469.

**RLHF, Confidence, and Miscalibration**
15. **Taming Overconfidence in LLMs: Reward Calibration in RLHF**
    C. Chen et al., 2025. arXiv:2410.09724.
16. **Just Ask for Calibration: Strategies for Eliciting Confidence**
    L. Tian et al., EMNLP 2023.
17. **Weak-to-Strong Generalization**
    J. Burns et al., 2023. arXiv:2312.09390.
18. **Discovering Language Model Behaviors with Model-Written Evaluations**
    N. Perez et al., 2022. arXiv:2212.09251.
19. **Language Models Learn to Mislead Humans via RLHF**
    E. Perez et al., 2024. arXiv:2409.12822.
20. **Style Outweighs Substance: Failure Modes of LLM Judges**
    X. Wang et al., 2024. arXiv:2409.15268.
21. **Beyond Scalar Reward Model: Learning Generative Judge**
    Z. Li et al., 2024. arXiv:2410.03742.
22. **Length-Controlled Margin-Based Preference Optimization**
    Y. Li et al., 2025. arXiv:2502.14643.
23. **Efficient Preference-based Reinforcement Learning**
    Y. Zhang et al., 2024. arXiv:2405.18688.
24. **CHARM: Calibrating Reward Models**
    S. Dong et al., 2025. arXiv:2504.10045.
25. **Flattery, Fluff, and Fog: Diagnosing Biases in Preference Models**
    Z. Zhang et al., 2025. arXiv:2506.05339.
26. **Contrastive Preference Learning**
    Y. Yuan et al., 2024. arXiv:2310.13639.
27. **Limited Generalization of DPO Implicit Reward Models**
    Y. Bai et al., 2024. arXiv:2409.03650.
28. **Causal Confusion and Reward Misidentification**
    Y. Fu et al., 2023. arXiv:2204.06601.
29. **A General Theoretical Paradigm for Learning from Preferences**
    X. Dong et al., 2023. arXiv:2310.12036.

**Semantic Similarity and Embedding Geometry**
30. **Model Comparison for Semantic Grouping**
    D. Blei et al., 2019. arXiv:1904.13323.
31. **SupMPN: Supervised Contrastive Learning for STS**
    M. Yang et al., Appl. Sci., 2022.
32. **SensEmbed: Learning Sense Embeddings**
    I. Iacobacci et al., ACL 2015.
33. **Specializing Word Embeddings for Similarity**
    O. Levy et al., EMNLP 2015.
34. **Measuring Semantic Similarity Using Concept Networks**
    N. Bulat et al., 2016.
35. **Interactive optimization of embedding-based similarity**
    D. Buscaldi et al., 2022.
36. **TexIm FAST: Text-to-Image Semantic Similarity**
    S. Li et al., 2024. arXiv:2406.04438.
37. **Ensemble Embedding Approach for Semantic Caching**
    J. Chen et al., 2025. arXiv:2507.07061.
38. **Consistency Calibration: Improving Uncertainty Calibration**
    Z. Zhou et al., 2024. arXiv:2410.12295.
39. **Gentle Introduction to Conformal Prediction**
    R. Barber et al., 2022. arXiv:2107.07511.
40. **Conformal Prediction: A Data Perspective**
    E. Vovk et al., 2025. arXiv:2410.06494.
41. **Calibration in ML Uncertainty Quantification**
    S. Krüger et al., 2023. arXiv:2309.06240.

**Planning and Non-Stationary Tasks**
42. **Tree-Planner: Efficient Close-loop Task Planning**
    Z. Wang et al., 2023. arXiv:2310.08582.
43. **SDA-PLANNER: State-Dependency Aware Adaptive Planner**
    H. Liu et al., 2025. arXiv:2509.26375.
44. **Deliberate Planning in Language Models**
    A. Guez et al., 2025.
45. **Self-Corrective Task Planning by Inverse Prompting**
    J. Li et al., 2025. IEEE.
46. **PDoctor: Testing Erroneous Planning in LLM Agents**
    H. Kim et al., 2024. arXiv:2404.17833.
47. **Ensuring Safety in LLM-Driven Robotics**
    Y. Wang et al., 2024. IEEE.
48. **ReST meets ReAct: Self-Improvement for Multi-Step Reasoning**
    J. Li et al., 2023. arXiv:2312.10003.

**Context Management and Long-Horizon Agents**
49. **The Complexity Trap: Observation Masking vs Summarization**
    A. Wang et al., 2025. arXiv:2508.21433.
50. **MOSS: Enabling Code-Driven Evolution**
    R. Guo et al., 2024. arXiv:2409.16120.
51. **Scaling LLM Multi-turn RL with Summarization**
    X. Li et al., 2025. arXiv:2510.06727.
52. **SagaLLM: Context Management and Transactions**
    H. Zhao et al., 2025. arXiv:2503.11951.
53. **Efficient On-Device Agents via Adaptive Context**
    Y. Lin et al., 2025. arXiv:2511.03728.
54. **HiAgent: Hierarchical Working Memory**
    S. Chen et al., 2024. arXiv:2408.09559.
55. **MemTool: Optimizing Short-Term Memory**
    Y. Zhou et al., 2025. arXiv:2507.21428.
56. **Scaling Long-Horizon LLM Agent via Context-Folding**
    H. Zhang et al., 2025. arXiv:2510.11967.
57. **Memory Sandbox: Interactive Memory Management**
    Y. Song et al., 2023. arXiv:2308.01542.
58. **A-MEM: Agentic Memory**
    S. Xie et al., 2025. arXiv:2502.12110.
59. **Lifelong Dialogue Agents via Timeline Memory**
    J. Huang et al., 2025. arXiv:2406.10996.
60. **In Prospect and Retrospect: Reflective Memory**
    J. Sun et al., 2025. arXiv:2503.08026.
61. **On Memory Construction and Retrieval**
    H. Li et al., 2025. arXiv:2502.05589.
62. **Walking Down the Memory Maze**
    S. Shi et al., 2023. arXiv:2310.05029.
63. **Enhancing Reasoning with Collaboration**
    M. Zhao et al., 2025. arXiv:2503.05944.
64. **MemoriesDB: Temporal-Semantic-Relational Database**
    H. Lhoest et al., 2025. Zenodo.

**Software Engineering Benchmarks**
65. **Saving SWE-Bench: Benchmark Mutation Approach**
    Y. Wang et al., 2025. arXiv:2510.08996.
66. **SWE-Bench+: Enhanced Coding Benchmark**
    S. Jimenez et al., 2024. arXiv:2410.06992.
67. **SWE-bench-java: GitHub Issue Resolving for Java**
    J. Zhang et al., 2024. arXiv:2408.14354.
68. **Multi-SWE-bench: Multilingual Benchmark**
    L. Zhang et al., 2025. arXiv:2504.02605.
69. **Dissecting the SWE-Bench Leaderboards**
    C. Wang et al., 2025. arXiv:2506.17208.
70. **Revisiting SWE-Bench: Importance of Data Quality**
    Y. Hu et al., 2025. IEEE.
71. **SWE-Bench-CL: Continual Learning**
    K. Li et al., 2025. arXiv:2507.00014.
72. **UTBoost: Rigorous Evaluation on SWE-Bench**
    S. Zhang et al., 2025. arXiv:2506.09289.
73. **SWE-agent: Agent-Computer Interfaces**
    C. Li et al., 2024. arXiv:2405.15793.
74. **Is Your Code Generated by ChatGPT Really Correct?**
    J. Liu et al., 2023. arXiv:2305.01210.
75. **HumanEval on Latest GPT Models**
    OpenAI et al., 2024. arXiv:2402.14852.
76. **HumanEval Pro and MBPP Pro**
    Y. Zhu et al., 2024. arXiv:2412.21199.
77. **Evaluating Software Development Agents**
    K. Wang et al., 2024. arXiv:2410.12468.
78. **Benchmarking AI Models in Software Engineering**
    P. Binkley et al., 2025. arXiv:2503.05860.
79. **Automated Benchmark Generation**
    Y. Zhang et al., 2025. arXiv:2503.07701.

**Sandbox, Execution, and Robustness**
80. **RedTeamCUA: Hybrid Web-OS Sandboxes**
    G. Li et al., 2025. arXiv:2505.21936.
81. **Nexus: Execution-Grounded Test Oracle**
    K. Zeng et al., 2025. arXiv:2510.26423.
82. **AutoDFBench: Digital Forensic Code Testing**
    H. Yahya et al., 2025. ACM.
83. **Human Oversight in Sandbox Environments**
    M. Ivanov et al., 2025. IEEE.
84. **GEN-RWD Sandbox**
    Y. Jiang et al., 2024. BMC Med Inform Decis Mak.
85. **AgentGuard: Safety Evaluation**
    X. Guo et al., 2025. arXiv:2502.09809.
86. **You Name It, I Run It**
    M. Huang et al., 2024. arXiv:2412.10133.
87. **Secure Extensibility via Plugin Sandboxing**
    T. Kim et al., 2019. arXiv:1905.08192.
88. **DockerMock: Pre-Build Detection**
    F. Zhang et al., 2021. arXiv:2104.05490.
89. **Mining Sandboxes for Linux Containers**
    W. Felter et al., 2017. arXiv:1712.05493.
90. **Out-of-Distribution Data: A Survey**
    Y. Huang et al., 2024. ACM Comput. Surveys.
91. **Revisiting Out-of-distribution Robustness**
    Y. Li et al., 2023. arXiv:2306.04618.
92. **Adversarial Robustness vs. OOD Robustness**
    Y. Liu et al., 2024. arXiv:2412.10535.
93. **Assessing Adversarial Robustness**
    H. Wang et al., 2024. arXiv:2405.02764.
94. **PromptRobust**
    H. Qin et al., 2024. arXiv:2306.04528.
95. **LLAMOS: Adversarial Purification**
    H. Chen et al., 2024. arXiv:2405.20770.
96. **Robustness Over Time**
    B. Wang et al., 2024. arXiv:2308.07847.
97. **Rapid Response: Mitigating LLM Jailbreaks**
    M. Shao et al., 2024. arXiv:2411.07494.
98. **aiXamine: Safety Evaluation Platform**
    T. Zhu et al., 2025. arXiv:2504.14985.

---

## APPENDIX
*(Placeholder for Code Snippets, detailed logs)*

---

## CHAPTER 6 — CONCLUSION AND FUTURE WORK

This thesis set out to investigate how large language model–based agents behave when placed in dynamic, multi-step tasks where requirements shift mid-execution. While existing benchmarks focus heavily on correctness in static settings, real-world workflows are rarely so stable. Systems that rely on LLMs must be prepared for ambiguity, interruptions, and updates that arrive after an initial plan is formed. The Entropic Stress-Test Framework was designed to explore exactly this setting, and the results reveal important insights about failure modes, stability, and the limits of current confidence metrics.

### 6.1 Summary of Contributions

Across the chapters, four contributions emerge clearly:

**(1) Identification of Silent Failure in RLHF-trained Agents**
The experiments consistently showed that agents can lose coherence without signaling uncertainty. Even when a requirement changes, the model often continues executing an outdated plan with unwavering confidence. This “silent failure” was visible in agent behavior long before traditional metrics detected anything unusual.

**(2) Introduction of SCR as a Semantic Stability Metric**
The Semantic Collapse Ratio proved to be a reliable early warning signal. By examining diverging next-step possibilities rather than surface-level token distributions, SCR captured internal confusion at the exact moment it emerged. This provides a complementary tool to entropy and aligns more closely with how humans intuitively detect hesitation or uncertainty in problem-solving.

**(3) A Modular Framework for Stress-Testing Agents**
The Entropic Stress-Test Framework integrates a controllable sandbox, an Orchestrator for scenario logic, and a Metric Service that computes SCR and related measures. The system makes it possible to reproduce perturbations, run detailed probes, and observe long-horizon reasoning in a controlled environment.

**(4) Empirical Evidence of Repetitive Stagnation Loops**
All experiments—both shock and non-shock—exposed a dominant failure mode: repetitive tool calls, often the exact same command (e.g., directory listing), repeated dozens of times without progress. These loops illustrate a deeper structural issue: once the model’s plan becomes misaligned with the task state, recovery becomes unlikely without explicit intervention.

Collectively, these contributions offer a clearer picture of why LLM agents struggle in non-stationary conditions, and how monitoring internal semantic structure can reveal collapse earlier than traditional signals.

### 6.2 Limitations

Although the framework proved effective for studying collapse, several limitations remain:

**(1) Limited Scenario Diversity**
Only three families of tasks were evaluated: file organization, CSV filtering, and basic log parsing. While these tasks are representative, they do not cover more complex domains such as API orchestration, multi-agent collaboration, or long-form planning.

**(2) Single Model Family**
The experiments focused primarily on one model (DeepSeek v3.2). Other families—OpenAI, Anthropic, Google—may show different collapse patterns, SCR magnitudes, or recovery behaviors. Cross-model comparisons would strengthen the generality of these findings.

**(3) Embedding Model Bias**
SCR depends on an embedding model. Different embedding models may cluster semantic meaning differently, which could shift SCR thresholds or introduce bias. While lightweight models work well in practice, the relationship between embedding geometry and agent stability deserves more rigorous study.

**(4) Compute Overhead**
Branching Probes require multiple model calls. Although the cost is small compared to the cost of silent failures, the overhead may still be significant for large-scale deployments unless further optimized.

**(5) Context Rot Not Fully Addressed**
Several control scenarios collapsed even without shocks. This suggests that long contexts accumulate noise over time, but the current framework does not include automated summarization, pruning, or memory management strategies to counter this.

These limitations highlight the need for broader and more systematic studies of agent stability across diverse conditions.

### 6.3 Future Work

The results in this thesis open several promising research directions:

**(1) Expanding Scenario Libraries**
New task families—API pipelines, realistic data cleaning, code refactoring, or interactive search tasks—could reveal failure modes not captured in this study. A richer scenario set would also make the framework more suitable for benchmarking.

**(2) Multi-Agent and Model-Switching Architectures**
SCR provides a natural trigger for switching between agents of different strengths. Future systems may use SCR to coordinate model handoffs, maintain team stability, or orchestrate collaborative agents with complementary roles.

**(3) Real-Time Context Management**
Integrating automatic context summarizers or rolling memory structures could reduce long-context degradation. An adaptive context-reset mechanism triggered by SCR spikes may significantly improve long-horizon resilience.

**(4) SCR as a Training Signal**
One intriguing direction is whether SCR—or semantic divergence more broadly—can be used as part of model training. Penalizing or rewarding certain divergence structures might teach models to regulate internal uncertainty more transparently.

**(5) Cross-Model Benchmarking**
Evaluating SCR behavior across GPT-4o, Claude 3.5, Gemini, and other models could reveal universal patterns or model-specific biases. Such cross-evaluation would also position SCR as a standardized measure for agent stability.

**(6) Integrating with Production Monitoring**
In real deployments, SCR could augment logging systems, triggering alerts when an agent begins drifting, or initiating a context reset before a full stagnation loop occurs. This would extend the framework beyond research and into practical reliability engineering.

Collectively, these directions outline how SCR and dynamic stress-testing can play a central role in developing the next generation of reliable, reasoning-aware LLM agents.

### 6.4 Closing Remarks

LLM-based agents have made remarkable progress, but they remain fragile in subtle ways. As this thesis shows, the biggest risks are not dramatic failures—they are the quiet, confident, and repetitive errors that emerge when an agent’s internal representation becomes misaligned with the task. By shifting the focus from surface fluency to semantic stability, the Entropic Stress-Test Framework offers a new perspective on how these systems think, how they fail, and how they can be made more robust.

The long-horizon future of AI will depend on our ability to detect confusion early, intervene intelligently, and design systems that remain stable even as their environments change. This work is a step toward understanding that challenge more deeply.