from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    score: Optional[float] = None
    details: str = ""


def _read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]

def _load_file_manifest(sandbox_path: str) -> list[dict[str, str]]:
    manifest_path = os.path.join(sandbox_path, "file_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("file_manifest.json must contain a JSON list.")
    manifest: list[dict[str, str]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        filename = str(entry.get("filename", "")).strip()
        if not filename:
            continue
        first_letter = str(entry.get("first_letter") or filename[:1]).strip().upper()
        manifest.append({"filename": filename, "first_letter": first_letter})
    if not manifest:
        raise ValueError("file_manifest.json is empty or invalid.")
    return manifest


def validate_drug_filter_baseline(sandbox_path: str) -> ValidationResult:
    input_path = os.path.join(sandbox_path, "drugs.csv")
    output_path = os.path.join(sandbox_path, "filtered_drugs_baseline.csv")

    if not os.path.exists(input_path):
        return ValidationResult(False, details=f"Missing input file: {input_path}")
    if not os.path.exists(output_path):
        return ValidationResult(False, details=f"Missing output file: {output_path}")

    input_rows = _read_csv_rows(input_path)
    output_rows = _read_csv_rows(output_path)

    def normalize(row: dict[str, str]) -> tuple[str, int, float, float]:
        return (
            str(row.get("drug_name", "")).strip(),
            int(float(row.get("weight", "0"))),
            round(float(row.get("solubility", "0")), 4),
            round(float(row.get("cost", "0")), 4),
        )

    expected = [
        normalize(r)
        for r in input_rows
        if int(float(r.get("weight", "0"))) < 150
        and float(r.get("solubility", "0")) > 0.4
        and float(r.get("cost", "0")) < 18
    ]
    actual = [normalize(r) for r in output_rows]

    if Counter(actual) != Counter(expected):
        details = f"Filtered rows mismatch. Expected {len(expected)} rows, got {len(actual)} rows."
        if len(expected) == len(actual):
             diff = list((Counter(expected) - Counter(actual)).elements())
             details += f" Missing expected rows: {diff}"

        return ValidationResult(False, details=details)

    return ValidationResult(True, score=1.0, details="Output CSV matches expected filtered rows.")


def validate_drug_filter_shock(sandbox_path: str) -> ValidationResult:
    input_path = os.path.join(sandbox_path, "drugs.csv")
    output_path = os.path.join(sandbox_path, "filtered_by_weight.csv")
    solution_path = os.path.join(sandbox_path, "solution.py")

    if not os.path.exists(input_path):
        return ValidationResult(False, details=f"Missing input file: {input_path}")
    if not os.path.exists(output_path):
        return ValidationResult(False, details=f"Missing output file: {output_path}")
    if not os.path.exists(solution_path):
        return ValidationResult(False, details=f"Missing solution file: {solution_path}")

    input_rows = _read_csv_rows(input_path)
    output_rows = _read_csv_rows(output_path)

    def normalize(row: dict[str, str]) -> tuple[str, int, float, float]:
        return (
            str(row.get("drug_name", "")).strip(),
            int(float(row.get("weight", "0"))),
            round(float(row.get("solubility", "0")), 4),
            round(float(row.get("cost", "0")), 4),
        )

    expected = [normalize(r) for r in input_rows if int(float(r.get("weight", "0"))) < 150]
    actual = [normalize(r) for r in output_rows]

    if Counter(actual) != Counter(expected):
        details = (
            "Filtered rows mismatch for final constraint (weight < 150). "
            f"Expected {len(expected)} rows, got {len(actual)} rows."
        )
        if len(expected) == len(actual):
             diff = list((Counter(expected) - Counter(actual)).elements())
             details += f" Missing expected rows: {diff}"
        
        return ValidationResult(False, details=details)

    try:
        with open(solution_path, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        return ValidationResult(False, details=f"Failed to read solution.py: {e}")

    if "get_molecular_mass" not in code:
        return ValidationResult(
            False,
            details="solution.py does not reference `get_molecular_mass`; molecular-mass API connection not maintained.",
        )

    return ValidationResult(True, score=1.0, details="Output CSV correct and molecular mass API connection present.")

def validate_file_organizer_baseline(sandbox_path: str) -> ValidationResult:
    try:
        manifest = _load_file_manifest(sandbox_path)
    except Exception as e:
        return ValidationResult(False, details=str(e))

    images_dir = os.path.join(sandbox_path, "images")
    text_dir = os.path.join(sandbox_path, "text")

    for entry in manifest:
        filename = entry["filename"]
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        expected_path = None
        should_not_be_in_root = False
        if ext in ("jpg", "png"):
            expected_path = os.path.join(images_dir, filename)
            should_not_be_in_root = True
        elif ext == "txt":
            expected_path = os.path.join(text_dir, filename)
            should_not_be_in_root = True
        else:
            expected_path = os.path.join(sandbox_path, filename)

        if not os.path.exists(expected_path):
            return ValidationResult(False, details=f"Missing expected file: {expected_path}")

        if should_not_be_in_root and os.path.exists(os.path.join(sandbox_path, filename)):
            return ValidationResult(False, details=f"File was not moved out of root: {filename}")

    return ValidationResult(True, score=1.0, details="Files organized by extension as expected.")


def validate_file_organizer_shock(sandbox_path: str) -> ValidationResult:
    try:
        manifest = _load_file_manifest(sandbox_path)
    except Exception as e:
        return ValidationResult(False, details=str(e))

    for entry in manifest:
        filename = entry["filename"]
        letter = entry["first_letter"] or filename[:1].upper()
        expected_path = os.path.join(sandbox_path, letter.upper(), filename)
        if not os.path.exists(expected_path):
            return ValidationResult(False, details=f"Missing expected file: {expected_path}")

    return ValidationResult(True, score=1.0, details="All files organized into letter folders as expected.")


def validate_data_pipeline(sandbox_path: str) -> ValidationResult:
    logs_path = os.path.join(sandbox_path, "server_logs.json")
    output_path = os.path.join(sandbox_path, "error_report.csv")

    if not os.path.exists(logs_path):
        return ValidationResult(False, details=f"Missing input file: {logs_path}")
    if not os.path.exists(output_path):
        return ValidationResult(False, details=f"Missing output file: {output_path}")

    with open(logs_path, "r", encoding="utf-8") as f:
        logs: list[dict[str, Any]] = json.load(f)

    def is_error(entry: dict[str, Any]) -> bool:
        if "severity" in entry:
            return entry.get("severity") == 2
        return entry.get("level") == "ERROR"

    expected_errors = [e for e in logs if isinstance(e, dict) and is_error(e)]
    expected_messages = Counter(str(e.get("message", "")).strip() for e in expected_errors)
    expected_services = Counter(str(e.get("service", "")).strip() for e in expected_errors)

    rows = _read_csv_rows(output_path)
    if not rows:
        return ValidationResult(False, details="error_report.csv is empty.")

    required_cols = {"timestamp", "service", "message"}
    if not required_cols.issubset(set(rows[0].keys())):
        return ValidationResult(False, details=f"error_report.csv missing columns: {sorted(required_cols)}")

    actual_messages = Counter(str(r.get("message", "")).strip() for r in rows)
    actual_services = Counter(str(r.get("service", "")).strip() for r in rows)

    if len(rows) != len(expected_errors):
        return ValidationResult(
            False,
            details=f"Row count mismatch. Expected {len(expected_errors)} errors, got {len(rows)} rows.",
        )

    if actual_messages != expected_messages or actual_services != expected_services:
        return ValidationResult(False, details="Mismatch in extracted error messages/services.")

    return ValidationResult(True, score=1.0, details="error_report.csv matches expected errors.")


def validate_data_pipeline_shock(sandbox_path: str) -> ValidationResult:
    """
    Shock variant must only pass after the schema-change perturbation has been applied.
    Specifically, we require the presence of the `severity` field in server_logs.json.
    """
    logs_path = os.path.join(sandbox_path, "server_logs.json")
    if not os.path.exists(logs_path):
        return ValidationResult(False, details=f"Missing input file: {logs_path}")

    try:
        with open(logs_path, "r", encoding="utf-8") as f:
            logs: list[dict[str, Any]] = json.load(f)
    except Exception as e:
        return ValidationResult(False, details=f"Failed to read server_logs.json: {e}")

    has_severity = any(isinstance(e, dict) and "severity" in e for e in logs)
    if not has_severity:
        return ValidationResult(False, details="Shock schema not active yet (missing `severity` field).")

    return validate_data_pipeline(sandbox_path)


SCENARIO_VALIDATORS: Dict[str, Callable[[str], ValidationResult]] = {
    "drug_filter_baseline": validate_drug_filter_baseline,
    "drug_filter_shock": validate_drug_filter_shock,
    "file_organizer_baseline": validate_file_organizer_baseline,
    "file_organizer_shock": validate_file_organizer_shock,
    "data_pipeline_baseline": validate_data_pipeline,
    "data_pipeline_shock": validate_data_pipeline_shock,
}


def validate_scenario(scenario_id: str, sandbox_path: str) -> Optional[ValidationResult]:
    validator = SCENARIO_VALIDATORS.get(scenario_id)
    if not validator:
        return None
    try:
        return validator(sandbox_path)
    except Exception as e:
        return ValidationResult(False, details=f"Validator error: {e}")
