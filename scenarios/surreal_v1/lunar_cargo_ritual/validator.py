from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    score: Optional[float] = None
    details: str = ""


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate(sandbox_path: str) -> ValidationResult:
    """
    Scenario-specific validator template.

    Replace placeholder paths/logic with deterministic checks.
    Keep this function pure: no network, no randomness.
    """
    output_path = os.path.join(sandbox_path, "output.json")
    expected_path = os.path.join(sandbox_path, "expected", "expected_output.json")

    if not os.path.exists(output_path):
        return ValidationResult(False, details=f"Missing required output: {output_path}")

    # Expected file is optional in scaffold stage.
    if not os.path.exists(expected_path):
        return ValidationResult(
            False,
            details=(
                "Validator scaffold not finalized: expected artifact missing. "
                f"Create {expected_path} and implement scenario checks."
            ),
        )

    try:
        actual = _read_json(output_path)
        expected = _read_json(expected_path)
    except Exception as e:
        return ValidationResult(False, details=f"Validator parse error: {e}")

    if actual != expected:
        return ValidationResult(False, score=0.0, details="Output mismatch against expected artifact.")

    return ValidationResult(True, score=1.0, details="Output matches expected artifact.")
