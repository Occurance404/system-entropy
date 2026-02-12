from __future__ import annotations

import re
from typing import Any, Dict, List


def agent_signaled_completion(content: Any) -> bool:
    if not isinstance(content, str):
        return False
    text = content.strip().lower()
    patterns = (
        r"^task(?:\s+is)?\s+complete\b",
        r"^completed\s+the\s+task\b",
        r"^mission\s+accomplished\b",
        r"^task_complete\s*[:=]\s*true\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def normalize_ai_verifier_result(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {"verdict": "uncertain", "confidence": 0.0, "reason": "invalid verifier response type"}

    verdict = str(raw.get("verdict", "uncertain")).strip().lower()
    if verdict not in ("done", "not_done", "uncertain"):
        verdict = "uncertain"

    confidence = raw.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))

    reason = str(raw.get("reason", "")).strip()
    if not reason:
        reason = "no reason provided"

    return {"verdict": verdict, "confidence": confidence, "reason": reason}


def run_ai_verifier(
    *,
    assess_func: Any,
    history: List[Dict[str, Any]],
    scenario_id: str,
    ground_truth_goal: str,
    sandbox_path: str,
    trigger: str,
) -> Dict[str, Any]:
    if not callable(assess_func):
        return {"verdict": "uncertain", "confidence": 0.0, "reason": "agent has no assess_completion()"}

    try:
        raw = assess_func(
            history=history,
            scenario_id=scenario_id,
            ground_truth_goal=ground_truth_goal,
            sandbox_path=sandbox_path,
            trigger=trigger,
        )
    except TypeError:
        # Backward-compatible fallback for simpler assess_completion(history) signatures.
        try:
            raw = assess_func(history)
        except Exception as e:
            return {"verdict": "uncertain", "confidence": 0.0, "reason": f"verifier error: {e}"}
    except Exception as e:
        return {"verdict": "uncertain", "confidence": 0.0, "reason": f"verifier error: {e}"}

    return normalize_ai_verifier_result(raw)
