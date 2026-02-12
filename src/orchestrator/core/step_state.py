from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional


def build_task_complete_payload(
    *,
    run_id: str,
    scenario_id: str,
    step_index: int,
    model_name: str,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "step_index": step_index,
        "model": model_name,
        "event_type": "task_complete",
        "task_complete": True,
        "timestamp": datetime.now().isoformat(),
    }


def build_default_step_metrics(
    *,
    run_id: str,
    scenario_id: str,
    step_index: int,
    model_name: str,
    panic_counter: int,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "step_index": step_index,
        "model": model_name,
        "current_entropy": None,
        "ige": None,
        "scr": None,
        "cbf": None,
        "rdi": None,
        "compression_ratio": None,
        "event_type": None,
        "panic_counter": panic_counter,
        "tool": None,
        "task_complete": None,
        "agent_done_claimed": False,
        "validation_passed": None,
        "validation_score": None,
        "validation_details": None,
        "ai_verifier_verdict": None,
        "ai_verifier_confidence": None,
        "ai_verifier_reason": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "timestamp": datetime.now().isoformat(),
    }


def check_perturbation_trigger(scenario: Dict[str, Any], step_index: int) -> Optional[str]:
    for perturbation in scenario.get("perturbations", []):
        if perturbation["step"] == step_index:
            return perturbation["instruction"]
    return None


def compose_validation_details_for_log(step_metrics: Dict[str, Any]) -> Optional[str]:
    details = step_metrics.get("validation_details")
    verifier = step_metrics.get("ai_verifier_verdict")
    if not verifier:
        return details
    reason = step_metrics.get("ai_verifier_reason") or ""
    confidence = step_metrics.get("ai_verifier_confidence")
    verifier_summary = f"ai_verifier={verifier}; confidence={confidence}; reason={reason}"
    if details:
        return f"{details} | {verifier_summary}"
    return verifier_summary
