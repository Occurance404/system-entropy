# Annexure - 1 Application Form

**Student Details:**

| Field | Value |
| :--- | :--- |
| **Name of Student:** | [Your Name] |
| **Registration No:** | [Your Reg No] |
| **Program:** | [e.g., B.Tech CSE / M.Tech AI] |
| **School:** | SEAS (School of Engineering and Applied Sciences) |
| **CGPA:** | [Leave blank/Fill] |
| **Any Backlog:** | Yes ☐ No ☑ |

---

**1. Title of the Project:**
**Quantifying Cognitive Collapse: An Entropic Framework for Stress-Testing and Rescuing Autonomous AI Agents**

**2. Problem Statement:**
While Large Language Models (LLMs) demonstrate high proficiency in short-term tasks, they exhibit a phenomenon known as "Cognitive Collapse" or "Context Rot" during long-horizon, multi-step workflows. Current benchmarks measure binary success/failure but fail to quantify *how* an agent's reasoning degrades over time or when subjected to environmental "shocks" (e.g., changing requirements mid-task). There is a critical lack of granular metrics to detect the onset of this degradation (looping, hallucination, goal drift) before catastrophic failure occurs.

**3. Objectives:**
1.  **Develop a Novel Metric Suite:** To validate and standardize two new metrics: *Semantic Collapse Ratio (SCR)* for measuring agent confusion and *Regressive Debt Index (RDI)* for measuring goal drift.
2.  **Stress-Test SOTA Models:** To subject leading models (GPT-4o, Claude 3.5 Sonnet, Mistral Large) to "Entropic Shock" experiments to determine their breaking points.
3.  **Implement a Rescue Protocol:** To demonstrate an automated "Handoff Mechanism" that detects high-entropy states (panic) and switches control to a superior model to salvage the task.

**4. Methodology:**
1.  **Framework Development:** We have developed a Python-based `Orchestrator` engine that manages agent interactions within a Dockerized sandbox. This engine includes a "State Monitor" that calculates entropy and embedding distances in real-time.
2.  **Experiment Design (The "Shock"):** We utilize a "Drug Discovery Filter" scenario where the agent is given a specific goal, and at Step N (The Perturbation), the requirements are subtly altered.
3.  **Data Collection (High-Cost Phase):** We will execute simulation runs across different model providers. For every step, we use a "Branching Probe" technique (generating 5 parallel "thoughts") to calculate the Semantic Collapse Ratio. This requires significant API inference volume.
4.  **Analysis:** We will correlate the SCR and RDI metrics with task success rates to prove their predictive validity.

**5. Outcome:**
1.  **Publication:** A research paper targeting a Scopus-indexed conference (e.g., AAAI, IJCAI, or an IEEE Transaction on AI) detailing the "Cognitive Collapse" phenomenon.
2.  **Dataset:** A release of the `entropic-shock` dataset containing detailed interaction logs of agents under stress.
3.  **Open Source Tool:** The simulation framework (already prototyped) will be released to the research community.

**Budget in detail:**

| Item | Description | Amount in INR |
| :--- | :--- | :--- |
| **Field data collection** | **Generation of Synthetic Interaction Logs:** Cost of running extensive "Branching Probes" (5x parallel inference per step) across simulation runs to generate the statistical dataset required for the paper. (Using OpenAI/Anthropic APIs). | 18,000 |
| **Characterization/User Charge** | **Inference & Compute Charges:** Costs associated with accessing proprietary "Reasoning Models" (e.g., OpenAI o1 or GPT-4o) to serve as the "Ground Truth" and "Rescue Agent" for the experiments. | 7,000 |
| **Any other** | N/A | - |
| **Total** | | **25,000** |

*(Note: Applying as Individual Student - Category A).*

**Availed Any Financial Support from SRM University AP:** Yes ☐ No ☑
**Provide Details (if yes):** N/A

---
*(Copy of the Proposal attached)*