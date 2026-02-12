# Project Vision: The Context-Aware Reliability Layer

## 1. The Core Philosophy
**"Infrastructure over Intelligence."**

We don't need a smarter model; we need a stricter manager. Current coding agents (like SWE-agent, OpenHands, etc.) fail not because they can't code, but because they lose track of reality (**Context Drift**). They forget where they are, what they broke, and why they made previous decisions.

Our goal is not to build another agent from scratch, but to build the **"Brain" and "Nervous System"** that makes *any* agent reliable.

## 2. The Solution: A Two-Part System
We will clone a SOTA agent and integrate two critical components:

### A. The Context Engine (The "Lens")
*   **The Problem:** Agents are overwhelmed by noise. They see too much (irrelevant files) or too little (missing error logs). Simple vector search is not enough.
*   **The Fix: Dynamic Retrieval & Ranking.**
    *   **Ranking System:** Every piece of context is assigned a "Priority Score" based on the current state.
        *   **P0 (Critical):** The exact error message or `stderr` causing the crash. **(Never Truncated)**
        *   **P1 (Evidence):** The stack trace and the specific lines of code referenced.
        *   **P2 (Action):** The `git diff` of what was just changed.
        *   **P3 (Relational):** Hard links (imports, definitions) connected to the modified code.
        *   **P4 (Soft):** Semantic search results (embeddings) and general history.
    *   **Token Budget Manager:** We treat the context window like a financial budget. If the window is full, we aggressively drop P4 and P3 context to ensure P0 and P1 are *always* perfectly visible.

### B. The Infrastructure Manager (The "Safety Net")
*   **The Problem:** Agents spiral. They make a mistake, try to fix it, make it worse, and delete the project.
*   **The Fix: The "Adult in the Room."**
    *   **Transaction Semantics:** Every step is a git commit. The Manager acts as a wrapper around the agent.
    *   **Auto-Rollback:** If an agent's action causes a regression (tests fail, syntax error), the Manager instantly reverts the change and feeds the error back to the agent.
    *   **Loop Detection:** If the Manager detects the agent engaging in "Edit Churn" (changing the same lines back and forth) or "Retry Loops" (running the same failing command), it intervenes.
    *   **Intervention:** The Manager can trigger a "Driver Swap" (e.g., switching from a cheaper model to GPT-4/Opus) or halt execution to ask the user.

## 3. Implementation Plan
1.  **Base:** Clone a proven, extensible agent (e.g., OpenHands, Aider, or SWE-agent).
2.  **Inject:** Replace the default "Context Builder" or "Prompt/Memory" module with our **Context Engine**.
3.  **Wrap:** Wrap the main execution loop with our **Infrastructure Manager** to handle git snapshots and rollbacks.
