# ARCHITECTURE V3: Full Recovery Context-Aware Developer

**Status:** Proposed | **Base:** V2 + Gap Analysis

---

## Reality Check (Repo Mapping)

This document describes a **V3 target architecture**. The current repository implements a **V2 baseline** stress-test harness (orchestrator + sandbox + metrics + scenario validators), plus a few reliability features to avoid blocked runs.

**Already implemented (V2 + no-block additions):**
- Orchestrator loop + scenario execution: `src/orchestrator/engine.py`
- Sandbox backend selection (Docker → local fallback): `SANDBOX_BACKEND=auto|docker|local` (`src/orchestrator/engine.py`, `src/connectors/local_connect.py`)
- Scenario setup/reset guardrails: `RESET_SANDBOX=1|0` (`src/scenarios/setup_ops.py`)
- Metrics + logging (SCR/CR/entropy proxy with logprobs sanity): `src/services/metrics.py`
- Logprobs request fallback (don’t fail runs if unsupported): `REQUEST_LOGPROBS=auto|off|on` and `PROXY_REQUEST_LOGPROBS=auto|off|on` (`src/agent/real_agent.py`, `src/llm_proxy.py`)
- Optional cost reduction for branching probes: `CHEAP_MODE=1` or `PROXY_SCR_MODE=off` (`src/llm_proxy.py`)
- Secrets policy (block/warn/off): `SECRETS_POLICY=block|warn|off` (`src/security/secrets.py`)

**Not implemented yet (this V3 doc proposes these):**
- Action Classifier (EDITED/REASONED/FAILED/LOOPED)
- Token Budget Manager (priority-based context packing)
- Graduated Recovery Controller (multi-level escalation)
- Failure Taxonomy Router
- Multi-File Coherence Index
- Semantic Drift Detector
- Pre-commit validator / regression-gated commits

---

## Component Overview

| Component | Zone | Purpose | Priority |
|-----------|------|---------|----------|
| Action Classifier | Infrastructure | Detect if agent actually acted or just talked | P1 |
| Token Budget Manager | Context Engine | Allocate context space by priority | P1 |
| Graduated Recovery | Infrastructure | 5-level escalating recovery instead of wipe-all | P2 |
| Failure Taxonomy Router | Validation | Route different failures to different actions | P2 |
| Multi-File Coherence | Context Engine | Track cross-file dependencies | P3 |
| Semantic Drift Detector | Context Engine | Catch logic inversions without tests | P3 |

---

## 1. Action Classifier

**Problem:** Agent says "I edited the file" but nothing actually changed.

**Input:** Agent response + tool calls + workspace diff

**Output:** One of:
- `EDITED` — Tool ran, diff produced
- `REASONED` — No tool call, just analysis
- `FAILED` — Tool attempted, error returned
- `LOOPED` — Response nearly identical to previous N

**Key logic:** Compare current response to last 3 responses using text similarity. If >85% similar → LOOPED.

---

## 2. Token Budget Manager

**Problem:** Context gets randomly truncated, sometimes losing critical errors.

**Input:** List of context items (errors, diffs, imports, search results)

**Output:** Prioritized subset that fits in budget

**Priority order (never truncate higher priority for lower):**
1. Error / stderr (up to 2000 tokens) — **NEVER dropped**
2. Stack trace (up to 1000 tokens)
3. Git diff (up to 1500 tokens)
4. Import graph (up to 1000 tokens)
5. Lexical search (up to 1000 tokens)
6. Embeddings (remainder)

**Rule:** Drop entire lower-priority categories before truncating higher ones.

---

## 3. Graduated Recovery Controller

**Problem:** "Wipe context after 3 fails" loses useful diagnostic info.

**5-level escalation:**

| Level | Trigger | Action |
|-------|---------|--------|
| 1 | 1st failure | Prune tool outputs older than 5 steps |
| 2 | 2nd failure | Summarize old reasoning into compact digest |
| 3 | 3rd failure | Full context wipe + fresh context pack |
| 4 | 4th failure | Swap to stronger model (rescue driver) |
| 5 | 5th failure or REGRESSION | Git rollback to last passing commit |

**Reset:** Success resets failure counter (but not level). Full task completion resets everything.

---

## 4. Failure Taxonomy Router

**Problem:** All failures treated the same way.

**Routes:**

| Failure Type | How Detected | Recovery Action |
|--------------|--------------|-----------------|
| Syntax Error | "SyntaxError" in stderr | Rollback + retry |
| Test Failure | Non-zero exit code | Add error to context |
| Coherence Fail | Coherence Index flags issue | Inject cross-file refs |
| Stagnation | Same error 3x in a row | Escalate recovery level |
| Loop Detected | Action Classifier says LOOPED | Wipe + fresh pack |
| Regression | Score dropped from previous | Git rollback |

---

## 5. Multi-File Coherence Index

**Problem:** Agent edits `utils.py`, breaks `main.py` that imports it.

**Tracks:**
- Function signatures (name, args, return type)
- Which files import which symbols
- Type annotations

**Detection:** If a function signature changes, check all files that import it. Flag if caller still uses old signature.

**Runs:** After each edit, before validation.

---

## 6. Semantic Drift Detector

**Problem:** Agent flips `>` to `<` — passes syntax, breaks logic.

**Detects:**
- Comparison inversions: `>` ↔ `<`, `>=` ↔ `<=`, `==` ↔ `!=`
- Boolean inversions: `and` ↔ `or`, `not` added/removed
- Off-by-one: `range(n)` → `range(n-1)`

**Method:** Pattern match on diff hunks. Optional: LLM sanity check for ambiguous cases.

---

## Data Flow

**Not linear — has branches and recovery loops:**

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                                                         │
                    ▼                                                         │
    ┌───────────────────────────┐                                             │
    │ 1. Orchestrator           │                                             │
    │    (requests context)     │                                             │
    └───────────┬───────────────┘                                             │
                ▼                                                             │
    ┌───────────────────────────┐                                             │
    │ 2. Token Budget Manager   │                                             │
    │    (assembles pack)       │                                             │
    └───────────┬───────────────┘                                             │
                ▼                                                             │
    ┌───────────────────────────┐                                             │
    │ 3. Agent Driver           │                                             │
    │    (generates response)   │                                             │
    └───────────┬───────────────┘                                             │
                ▼                                                             │
    ┌───────────────────────────┐                                             │
    │ 4. Action Classifier      │──────────┐                                  │
    │                           │          │                                  │
    └───────────┬───────────────┘          │                                  │
                │                          │                                  │
        EDITED? │              LOOPED/FAILED                                  │
                ▼                          │                                  │
    ┌───────────────────────────┐          │                                  │
    │ 5. Workspace              │          │                                  │
    │    (executes, diffs)      │          │                                  │
    └───────────┬───────────────┘          │                                  │
                ▼                          │                                  │
    ┌───────────────────────────┐          │                                  │
    │ 6. Coherence + Drift      │          │                                  │
    │    (checks consistency)   │          │                                  │
    └───────────┬───────────────┘          │                                  │
                ▼                          │                                  │
    ┌───────────────────────────┐          │                                  │
    │ 7. Validators             │          │                                  │
    │    (syntax, tests)        │          │                                  │
    └───────────┬───────────────┘          │                                  │
                │                          │                                  │
        ┌───────┴───────┐                  │                                  │
        │               │                  │                                  │
      PASS            FAIL ◄───────────────┘                                  │
        │               │                                                     │
        ▼               ▼                                                     │
    ┌────────┐   ┌───────────────────────────┐                                │
    │ Done   │   │ 8. Failure Router         │                                │
    │ (next  │   │    (classifies failure)   │                                │
    │ step)  │   └───────────┬───────────────┘                                │
    └────┬───┘               ▼                                                │
         │       ┌───────────────────────────┐                                │
         │       │ 9. Recovery Controller    │────────────────────────────────┘
         │       │    (escalates if needed)  │       (loops back to step 1)
         │       └───────────────────────────┘
         │
         └──────────────────────────────────────────────────────────────────────┐
                                                                                │
                                              (loops back to step 1 for next step)
```

**Key branches:**
- After Action Classifier: LOOPED/FAILED skips workspace, goes straight to Failure Router
- After Validators: PASS → next step, FAIL → Recovery
- After Recovery: loops back to step 1 with modified context/state

---

## Recovery Paths

Three explicit recovery arrows:

1. **Failure → Recovery Controller** — Any failure triggers graduated response
2. **Git Manager → Workspace** — Rollback restores last passing commit
3. **Orchestrator → Driver** — Hot-swap to stronger model

---

## Implementation Order

**Phase 1 (Quick wins):**
- Action Classifier — Simple, high signal
- Token Budget Manager — Stabilizes context

**Phase 2 (Core recovery):**
- Graduated Recovery Controller
- Failure Taxonomy Router

**Phase 3 (Advanced detection):**
- Multi-File Coherence Index
- Semantic Drift Detector

---

## Diagram

See: [architecture_v3.svg](./architecture_v3.svg)
