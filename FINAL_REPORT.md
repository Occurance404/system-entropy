# Entropic Stress-Test: Final Report
> **Date:** 26 November 2025
> **Status:** Complete & Verified

## 1. Executive Summary
This project successfully implemented and validated a framework for detecting **"Context Rot"** (Cognitive Collapse) in Autonomous AI Agents. 

Our key finding is the **"Silent Killer" Phenomenon**: State-of-the-art optimized models (like `kat-coder-pro`) often report **0.0 Entropy** (100% confidence) even while their internal logic is collapsing. Standard observability tools fail to detect this.

We introduced two novel metrics that successfully detected the rot where Entropy failed:
1.  **Semantic Collapse Ratio (SCR):** Measures the divergence of the agent's future plans (Branching Probe).
2.  **Compression Ratio (CR):** Monitors the structural health (repetitive looping) of the agent's output.

## 2. System Architecture
The system consists of three core components:
1.  **The Orchestrator:** A state machine that executes the agent and injects "Shocks" (conflicting requirements) at specific steps.
2.  **The Monitor (Probe):** A sidecar process that calculates real-time metrics (SCR, IGE, CR).
3.  **The Sandbox:** A Docker-based environment allowing the agent to execute real shell commands, file operations, and python scripts.

## 3. Key Experiments & Results

### Experiment A: The "Silent Rot" (Validated)
*   **Setup:** Agent tasked with filtering a drug database. At Step 4 and Step 7, conflicting requirements were injected.
*   **Observation:**
    *   **Entropy:** Flatline at 0.0 (Model claimed 100% confidence).
    *   **SCR:** Spiked from **0.19** to **0.24**.
*   **Conclusion:** The agent was becoming incoherent, but lying about its confidence. SCR successfully pierced the veil.

### Experiment B: Rate Limit Stress (Lesson Learned)
*   **Setup:** Attempted to use `google/gemini-2.0-flash-exp`.
*   **Outcome:** The "Branching Probe" (5x parallel requests) triggered API rate limits immediately.
*   **Lesson:** "Active Probing" requires high-throughput API access. For free-tier models, the probe itself can be a denial-of-service attack.

## 4. How to Use the Framework

### Prerequisites
*   Docker
*   Python 3.11+
*   OpenAI-compatible API Key (e.g., OpenRouter, vLLM)

### Running a Simulation
```bash
# 1. Configure Environment
cp .env.example .env
# Edit .env with your VLLM_API_KEY and MODEL_NAME

# 2. Run the Rescue Experiment
.venv/bin/python run_rescue_experiment.py --max_steps 15

# 3. Visualize Results
.venv/bin/python visualize_results.py
```
The results will be saved to `data/results/experiment_summary.png`.

## 5. Future Work (Next Steps)
*   **Scenario Expansion:** We have upgraded the agent to have full `run_shell` access. The next logical step is to create a "Repo Repair" scenario where the agent must run `git`, `pytest`, and fix bugs in a real codebase.
*   **Intervention Protocols:** Currently, we "Rescue" by switching models. Future work could implement "Context Surgery" (pruning the context window to remove the "rotting" segments).
