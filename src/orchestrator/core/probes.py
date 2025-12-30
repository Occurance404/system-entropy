from typing import Any, Dict, List, Optional


def run_branching_probe(
    *,
    agent: Any,
    metric_service: Any,
    history: List[Dict[str, Any]],
    probe_branch_count: int,
    perturbation_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes the Branching Probe via Agent and calculates SCR via Service.
    If perturbation_instruction is None, probes the current state silently.
    """
    if perturbation_instruction:
        probe_history = history + [{"role": "user", "content": perturbation_instruction}]
    else:
        probe_history = list(history)

    branches = agent.generate_multiple(probe_history, n=probe_branch_count)
    branch_texts = [b.get("content", "") for b in branches]
    probe_total_tokens = 0
    for b in branches:
        usage = b.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int):
            probe_total_tokens += usage["total_tokens"]

    scr = metric_service.calculate_scr(branch_texts)
    return {"scr": scr, "branches": branch_texts, "probe_total_tokens": probe_total_tokens}

