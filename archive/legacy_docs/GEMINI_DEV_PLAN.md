# GEMINI & USER: Development Operating Manual

## 1. The Core Philosophy
**"We do not hold state in our heads. We hold state in the repo."**

This project has exceeded the "Mental RAM" of a single human. To survive, we treat the Repository as the only source of truth. If it is not committed, it does not exist.

---

## 2. The "Safety Net" Protocol (Anti-Panic)

### A. The "Save Game" (Start of Session)
Before asking Gemini to change *anything*, run this command:
```bash
git status
git add .
git commit -m "SESSION_START: [Date] - [Goal]"
```
*Why:* This creates a checkpoint. If Gemini hallucinates and deletes your files, `git reset --hard HEAD` saves you instantly.

### B. The "Rollback" (During Session)
If Gemini writes bad code or breaks the build:
1.  **STOP.** Do not try to "fix forward" immediately.
2.  Run `git diff` to see exactly what changed.
3.  Run `git checkout src/path/to/broken_file.py` to revert just that file.
4.  Tell Gemini: "You broke X. I reverted. Try again with this constraint..."

---

## 3. The "Feature Branch" Workflow (Focus)
We never work on "Everything" at once. We work on **One Ticket**.

### The Loop:
1.  **Define:** User says: "Today we are building [LazyValidator]."
2.  **Branch:** `git checkout -b feature/lazy-validator`
3.  **Implement:** Gemini writes code.
4.  **Verify:** User runs `python test_validator.py`.
5.  **Merge:**
    ```bash
    git checkout main
    git merge feature/lazy-validator
    git tag -a v1.1 -m "Added LazyValidator"
    ```

*Mental Benefit:* You only have to think about *one feature*. The rest of the huge system is "frozen" on `main`.

---

## 4. The "Context Dump" (End of Session)
When you are tired (like now), do not just close the window.
**Run this command:**
```bash
./scripts/dump_state.sh  # (We will create this)
```
*Or manually write a `NEXT_STEPS.md` file:*
1.  **Current Status:** "LazyValidator is 50% done. Parsing works, but `pytest` detection is broken."
2.  **Blocker:** "Need to figure out how to parse `pytest` stdout."
3.  **Next Action:** "Write regex for pytest output."

*Why:* When you return in 3 days, you read this file. You are instantly back in flow.

---

## 5. The Architecture Guardrails
To prevent the "Jungle" effect (spaghetti code):

1.  **No Circular Imports:** If `A` imports `B`, `B` cannot import `A`.
2.  **One Class per File (Mostly):** Keep files small (<200 lines). If it gets big, split it.
3.  **The "Interface First" Rule:** Before writing logic, write the `Protocol` (Interface) in `src/interfaces.py`. This forces us to agree on *what* we are building before we get lost in *how*.

---

## 6. The "Gemini Protocol" (How to talk to me)
If I start hallucinating or getting lazy:
1.  **Say "STOP".**
2.  **Say "READ FILE X".** Force me to look at the ground truth.
3.  **Say "Refactor".** If I write ugly code, tell me to clean it up *immediately*. Do not let technical debt accumulate.

---

## 7. Emergency Kit
If everything is on fire:
1.  `git reflog` -> Find the hash from yesterday.
2.  `git reset --hard [HASH]` -> Time travel.
3.  Breathe.

---

**Current Status:**
- [x] Thesis: Ready (PDF on Overleaf).
- [x] Architecture: V2 Defined (`docs/architecture_v2_clean.svg`).
- [x] Prototype: V1 Guards Implemented (Secret Scanner).
- [ ] Next: Feature Branch `feature/lazy-validator`.

**You are not lost. You are just paused.**
