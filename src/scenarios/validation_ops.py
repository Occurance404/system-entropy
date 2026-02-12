from __future__ import annotations

from typing import Callable, Dict, Optional

from src.scenarios.validators import (
    ValidationResult,
    validate_archive_impossible_dates,
    validate_bureau_contradictory_forms,
    validate_city_duplicate_identities,
    validate_data_pipeline,
    validate_data_pipeline_shock,
    validate_dream_court_transcript,
    validate_drug_filter_baseline,
    validate_drug_filter_shock,
    validate_file_organizer_baseline,
    validate_file_organizer_shock,
    validate_lunar_cargo_ritual,
    validate_missing_axiom,
    validate_museum_renamed_species,
    validate_oracle_contract_amendment,
    validate_paradox_lab_protocol,
    validate_signal_mirror_logs,
)


SCENARIO_VALIDATORS: Dict[str, Callable[[str], ValidationResult]] = {
    "drug_filter_baseline": validate_drug_filter_baseline,
    "drug_filter_shock": validate_drug_filter_shock,
    "file_organizer_baseline": validate_file_organizer_baseline,
    "file_organizer_shock": validate_file_organizer_shock,
    "data_pipeline_baseline": validate_data_pipeline,
    "data_pipeline_shock": validate_data_pipeline_shock,
    "archive_impossible_dates": validate_archive_impossible_dates,
    "museum_renamed_species": validate_museum_renamed_species,
    "dream_court_transcript": validate_dream_court_transcript,
    "lunar_cargo_ritual": validate_lunar_cargo_ritual,
    "paradox_lab_protocol": validate_paradox_lab_protocol,
    "oracle_contract_amendment": validate_oracle_contract_amendment,
    "city_duplicate_identities": validate_city_duplicate_identities,
    "signal_mirror_logs": validate_signal_mirror_logs,
    "missing_axiom": validate_missing_axiom,
    "bureau_contradictory_forms": validate_bureau_contradictory_forms,
}


def validate_scenario(scenario_id: str, sandbox_path: str) -> Optional[ValidationResult]:
    validator = SCENARIO_VALIDATORS.get(scenario_id)
    if not validator:
        return None
    try:
        return validator(sandbox_path)
    except Exception as e:
        return ValidationResult(False, details=f"Validator error: {e}")


__all__ = [
    "ValidationResult",
    "validate_drug_filter_baseline",
    "validate_drug_filter_shock",
    "validate_file_organizer_baseline",
    "validate_file_organizer_shock",
    "validate_data_pipeline",
    "validate_data_pipeline_shock",
    "validate_archive_impossible_dates",
    "validate_museum_renamed_species",
    "validate_dream_court_transcript",
    "validate_lunar_cargo_ritual",
    "validate_paradox_lab_protocol",
    "validate_oracle_contract_amendment",
    "validate_city_duplicate_identities",
    "validate_signal_mirror_logs",
    "validate_missing_axiom",
    "validate_bureau_contradictory_forms",
    "SCENARIO_VALIDATORS",
    "validate_scenario",
]
