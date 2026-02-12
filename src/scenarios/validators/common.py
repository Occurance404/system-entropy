from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    score: Optional[float] = None
    details: str = ""


def _read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _require_columns(rows: list[dict[str, str]], required: set[str], file_name: str) -> Optional[ValidationResult]:
    if not rows:
        return ValidationResult(False, details=f"{file_name} is empty.")
    cols = set(rows[0].keys())
    if not required.issubset(cols):
        return ValidationResult(False, details=f"{file_name} missing columns: {sorted(required)}")
    return None


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
