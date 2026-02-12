from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Callable, Dict, Tuple, Any, Optional


def _apply_data_pipeline_schema_change(sandbox_path: str) -> None:
    """
    Mutates server_logs.json from the v1 schema:
      {timestamp: ISO str, level: INFO|...|ERROR, ...}
    to the v2 schema described in the perturbation:
      {timestamp: epoch int, severity: 1|2, ...}
    """
    logs_path = os.path.join(sandbox_path, "server_logs.json")
    if not os.path.exists(logs_path):
        return

    with open(logs_path, "r", encoding="utf-8") as f:
        logs: list[dict[str, Any]] = json.load(f)

    mutated: list[dict[str, Any]] = []
    for entry in logs:
        if not isinstance(entry, dict):
            continue

        level = entry.get("level")
        severity = 2 if level == "ERROR" else 1

        ts = entry.get("timestamp")
        epoch: Optional[int] = None
        if isinstance(ts, (int, float)):
            epoch = int(ts)
        elif isinstance(ts, str):
            try:
                epoch = int(datetime.fromisoformat(ts).timestamp())
            except Exception:
                epoch = None

        new_entry = dict(entry)
        new_entry.pop("level", None)
        new_entry["severity"] = severity
        if epoch is not None:
            new_entry["timestamp"] = epoch
        mutated.append(new_entry)

    with open(logs_path, "w", encoding="utf-8") as f:
        json.dump(mutated, f, indent=2)


def _apply_drug_filter_api_stub(sandbox_path: str) -> None:
    """
    Adds a local API stub + docs so the shock is actionable without network access.
    """
    docs_path = os.path.join(sandbox_path, "molecular_mass_api_docs.md")
    module_path = os.path.join(sandbox_path, "molecular_mass_api.py")

    # Lightweight docs
    if not os.path.exists(docs_path):
        with open(docs_path, "w", encoding="utf-8") as f:
            f.write(
                "# Molecular Mass API (Local Stub)\n"
                "\n"
                "Use `from molecular_mass_api import get_molecular_mass`.\n"
                "This simulates an external API without network calls.\n"
            )

    # Simple deterministic stub based on the provided drugs.csv fixture
    if not os.path.exists(module_path):
        with open(module_path, "w", encoding="utf-8") as f:
            f.write(
                "from __future__ import annotations\n"
                "\n"
                "from typing import Dict\n"
                "\n"
                "\n"
                "_MOCK_MASSES: Dict[str, float] = {\n"
                "    \"A\": 100.0,\n"
                "    \"B\": 150.0,\n"
                "    \"C\": 200.0,\n"
                "    \"D\": 120.0,\n"
                "    \"E\": 250.0,\n"
                "}\n"
                "\n"
                "\n"
                "def get_molecular_mass(drug_name: str) -> float:\n"
                "    \"\"\"Returns a deterministic mock molecular mass for known drugs.\"\"\"\n"
                "    return float(_MOCK_MASSES.get(str(drug_name).strip(), 0.0))\n"
            )


def _swap_file(sandbox_path: str, src_name: str, dst_name: str) -> None:
    src = os.path.join(sandbox_path, src_name)
    dst = os.path.join(sandbox_path, dst_name)
    if not os.path.exists(src):
        return
    with open(src, "r", encoding="utf-8") as f:
        payload = f.read()
    with open(dst, "w", encoding="utf-8") as f:
        f.write(payload)


def _apply_archive_dates_shock(sandbox_path: str) -> None:
    _swap_file(sandbox_path, "calendar_offsets_v2.json", "active_calendar_offsets.json")


def _apply_museum_taxonomy_shock(sandbox_path: str) -> None:
    _swap_file(sandbox_path, "taxonomy_v2.json", "active_taxonomy.json")


def _apply_dream_court_shock(sandbox_path: str) -> None:
    _swap_file(sandbox_path, "source_trust_v2.json", "active_source_trust.json")


def _apply_lunar_cargo_shock(sandbox_path: str) -> None:
    _swap_file(sandbox_path, "constraints_v2.json", "active_constraints.json")
    _swap_file(sandbox_path, "unit_profile_v2.json", "active_unit_profile.json")


def _apply_paradox_lab_shock(sandbox_path: str) -> None:
    _swap_file(sandbox_path, "lab_status_v2.json", "active_lab_status.json")


def _apply_oracle_contract_shock(sandbox_path: str) -> None:
    _swap_file(sandbox_path, "rules_v2.json", "active_rules.json")


def _apply_city_identity_shock(sandbox_path: str) -> None:
    _swap_file(sandbox_path, "identity_policy_v2.json", "active_identity_policy.json")


def _apply_signal_mirror_shock(sandbox_path: str) -> None:
    _swap_file(sandbox_path, "ordering_policy_v2.json", "active_ordering_policy.json")


def _apply_missing_axiom_shock(sandbox_path: str) -> None:
    _swap_file(sandbox_path, "axioms_v2.json", "active_axioms.json")


def _apply_bureau_policy_shock(sandbox_path: str) -> None:
    _swap_file(sandbox_path, "policy_v2.json", "active_policy.json")


PERTURBATION_OPS: Dict[Tuple[str, int], Callable[[str], None]] = {
    ("data_pipeline_shock", 4): _apply_data_pipeline_schema_change,
    ("drug_filter_shock", 4): _apply_drug_filter_api_stub,
    ("archive_impossible_dates", 4): _apply_archive_dates_shock,
    ("museum_renamed_species", 4): _apply_museum_taxonomy_shock,
    ("dream_court_transcript", 4): _apply_dream_court_shock,
    ("lunar_cargo_ritual", 4): _apply_lunar_cargo_shock,
    ("paradox_lab_protocol", 4): _apply_paradox_lab_shock,
    ("oracle_contract_amendment", 4): _apply_oracle_contract_shock,
    ("city_duplicate_identities", 4): _apply_city_identity_shock,
    ("signal_mirror_logs", 4): _apply_signal_mirror_shock,
    ("missing_axiom", 4): _apply_missing_axiom_shock,
    ("bureau_contradictory_forms", 4): _apply_bureau_policy_shock,
}


def apply_perturbation_if_needed(scenario_id: str, step_index: int, sandbox_path: str) -> None:
    op = PERTURBATION_OPS.get((scenario_id, step_index))
    if not op:
        return
    op(sandbox_path)
