# FUTURE ARCHITECTURAL BLUEPRINT: The "Context-Aware" Reliability Engine (V1)

## 1. The Pivot: From Thesis to Infrastructure
The original thesis proved that LLM Agents suffer from "Semantic Collapse" and "Stagnation" in long-horizon tasks.
The goal of this V1 architecture is not just to measure this failure, but to **fix it** by replacing the "Agent's Memory" with a deterministic "Infrastructure Brain."

**Core Philosophy:**
1.  **Trust Infrastructure, Not Intelligence:** Agents hallucinate path structures and forget context. The Infrastructure (Context Engine) must be the source of truth.
2.  **Evidence > Embeddings:** Context should be built from *Hard References* (Stack traces, imports, diffs), not fuzzy vector similarity.
3.  **Validation is Binary:** "Confidence" is irrelevant. Did the tests pass? Did the syntax parse?

---

## 2. The V1 Architecture

### A. The Validator (Ground Truth)
We avoid the friction of custom scoring scripts for every task.
*   **LazyValidator (Default):**
    *   Auto-detects and runs existing project tests: `pytest`, `npm test`, `cargo test`, `make test`.
    *   If no tests found: Runs "Smoke Checks" (Parse/Compile modified files).
    *   *Result:* Returns `PASS`, `FAIL`, or `REGRESSION` (score dropped).
*   **StrictValidator (Opt-in):**
    *   Uses a task-specific `score.py` or `eval_cmd` for benchmark-grade evaluation.
*   **NullValidator:**
    *   Pure refactor/cleanup tasks. Checks only for safety (no deleted files) and syntax.

### B. The Context Pack (The "Product")
Instead of dumping chat history, we construct a deterministic "Evidence Packet" for the agent at every step.
**Priority Order (Hard Refs > Soft Refs):**
1.  **Hard Evidence:**
    *   Last failing validator output (stderr).
    *   The specific stack trace (mapped to file:line).
    *   The last Git Diff (what did we just break?).
2.  **Hard Links:**
    *   Import graph edges from the modified/failing files.
    *   Definitions of symbols appearing in the stack trace.
3.  **Lexical Search (Deterministic):**
    *   `ripgrep` for error message keywords or function names.
4.  **Soft Search (Fallback):**
    *   Vector embeddings (only if the above yields zero results).

*Budget Rule:* Never truncate the critical bug. Drop low-priority items entirely rather than chopping the stack trace.

### C. The Driver Adapter (No Format Hell)
We treat the LLM (Codex, Gemini, DeepSeek) as a black box.
*   **Normalization:** We do not enforce a strict "JSON Patch" format that breaks weak models.
*   **Observation:** We calculate the "Action" by computing the `git diff` of the workspace after the agent runs.
*   **Benefit:** Swapping drivers (DeepSeek -> GPT-4) is trivial because we observe *outcomes*, not *formats*.

### D. Safety & Rollback (Transaction Semantics)
Agents destroy their own work. We add "Ctrl+Z" to the runtime.
1.  **Git Checkpoints:** Initialize a temporary git repo in the sandbox. Commit after every step.
2.  **Rollback Protocol:**
    *   If `Validator` returns `REGRESSION` (score went down) or `CRITICAL_FAIL` (syntax error):
    *   **Auto-Rollback** to the last passing checkpoint.
    *   Inform the agent: "Your last edit broke the build. I have reverted it. Here is the error."

### E. Collapse Triggers (The "Check Engine" Light)
We move beyond expensive SCR probes to cheap, real-time signals:
*   **Validator Regression:** Score drop.
*   **Edit Churn:** Rewriting the same lines 3+ times (detected via git diff).
*   **Repeated Errors:** Same stderr signature 3x in a row.
*   *Action:* Trigger "Soft Prune" (clear chat context) or "Escalate" (swap driver).

---

## 3. Implementation Roadmap (Shippable V1)

### Phase 1: The "Lazy" Foundation
- [ ] **Connector Upgrade:** Add `git init` to the sandbox. Implement `snapshot()` and `rollback()`.
- [ ] **LazyValidator:** Implement auto-detection for `pytest` and `python -m py_compile`.
- [ ] **Observer:** Replace "Tool Output" logging with "Git Diff" logging.

### Phase 2: The Context Engine (Static Analysis)
- [ ] **Python AST Extractor:** Simple script to parse imports and function defs (no LSP).
- [ ] **Regex Extractor:** Fallback for other languages.
- [ ] **Context Builder:** A module that takes `(error_log, modified_files)` and outputs `context_pack.txt`.

### Phase 3: The Orchestrator Logic
- [ ] **Loop Logic:** Run -> Checkpoint -> Validate -> (Commit or Rollback).
- [ ] **Wipe Policy:** If `Fail_Count > 3`, clear chat history, inject `context_pack.txt`.

---

## 4. Why This Wins
This architecture moves away from "Chatting with Code" to "Engineering with Code."
It is robust to model stupidity because the **Infrastructure** handles Memory, Verification, and Safety. The Model just proposes Diffs.
