from src.orchestrator.core.orchestrator import Orchestrator


def _orchestrator_stub():
    return object.__new__(Orchestrator)


def test_normalize_ai_verifier_result_valid():
    o = _orchestrator_stub()
    out = o._normalize_ai_verifier_result(
        {"verdict": "done", "confidence": "0.91", "reason": "tests passed"}
    )
    assert out["verdict"] == "done"
    assert out["confidence"] == 0.91
    assert out["reason"] == "tests passed"


def test_normalize_ai_verifier_result_invalid_type():
    o = _orchestrator_stub()
    out = o._normalize_ai_verifier_result("bad")
    assert out["verdict"] == "uncertain"
    assert out["confidence"] == 0.0


def test_normalize_ai_verifier_result_clamps_values():
    o = _orchestrator_stub()
    out = o._normalize_ai_verifier_result(
        {"verdict": "DONE", "confidence": 3.14, "reason": ""}
    )
    assert out["verdict"] == "done"
    assert out["confidence"] == 1.0
    assert out["reason"] == "no reason provided"
