from __future__ import annotations

import os
from collections import Counter
from itertools import combinations
from typing import Any

from .common import (
    ValidationResult,
    _read_csv_rows,
    _read_json,
    _require_columns,
    _to_float,
    _to_int,
)

def validate_archive_impossible_dates(sandbox_path: str) -> ValidationResult:
    events_path = os.path.join(sandbox_path, "timeline_events.csv")
    deps_path = os.path.join(sandbox_path, "dependencies.csv")
    offsets_path = os.path.join(sandbox_path, "active_calendar_offsets.json")
    output_path = os.path.join(sandbox_path, "timeline.csv")

    for path in (events_path, deps_path, offsets_path, output_path):
        if not os.path.exists(path):
            return ValidationResult(False, details=f"Missing required file: {path}")

    events = _read_csv_rows(events_path)
    deps = _read_csv_rows(deps_path)
    offsets = _read_json(offsets_path)
    actual_rows = _read_csv_rows(output_path)

    err = _require_columns(events, {"event_id", "event_name", "calendar", "base_day"}, "timeline_events.csv")
    if err:
        return err
    err = _require_columns(actual_rows, {"event_id", "event_name", "normalized_day"}, "timeline.csv")
    if err:
        return err
    err = _require_columns(deps, {"before_event_id", "after_event_id"}, "dependencies.csv")
    if err:
        return err

    expected = []
    for row in events:
        cal = str(row.get("calendar", "")).strip()
        if cal not in offsets:
            return ValidationResult(False, details=f"Unknown calendar '{cal}' in timeline_events.csv")
        normalized_day = _to_int(row.get("base_day")) + _to_int(offsets.get(cal))
        expected.append(
            {
                "event_id": str(row.get("event_id", "")).strip(),
                "event_name": str(row.get("event_name", "")).strip(),
                "normalized_day": normalized_day,
            }
        )
    expected.sort(key=lambda x: (x["normalized_day"], x["event_id"]))

    actual = []
    for row in actual_rows:
        actual.append(
            {
                "event_id": str(row.get("event_id", "")).strip(),
                "event_name": str(row.get("event_name", "")).strip(),
                "normalized_day": _to_int(row.get("normalized_day")),
            }
        )

    if len(actual) != len(expected):
        return ValidationResult(False, details=f"timeline.csv row count mismatch. Expected {len(expected)}, got {len(actual)}.")

    exp_t = [(r["event_id"], r["event_name"], r["normalized_day"]) for r in expected]
    act_t = [(r["event_id"], r["event_name"], r["normalized_day"]) for r in actual]
    if Counter(act_t) != Counter(exp_t):
        return ValidationResult(False, details="timeline.csv rows do not match expected normalized timeline.")

    # Accept tie-order flexibility when days are equal, but keep timeline monotonic.
    days = [r["normalized_day"] for r in actual]
    if any(days[i] > days[i + 1] for i in range(len(days) - 1)):
        return ValidationResult(False, details="timeline.csv is not sorted by normalized_day ascending.")

    pos = {row["event_id"]: idx for idx, row in enumerate(actual)}
    for dep in deps:
        before_id = str(dep.get("before_event_id", "")).strip()
        after_id = str(dep.get("after_event_id", "")).strip()
        if before_id not in pos or after_id not in pos:
            return ValidationResult(False, details=f"Dependency references unknown event(s): {before_id} -> {after_id}")
        if pos[before_id] >= pos[after_id]:
            return ValidationResult(False, details=f"Dependency order violated: {before_id} must be before {after_id}")

    return ValidationResult(True, score=1.0, details="timeline.csv matches normalized timeline and dependency order.")


def validate_museum_renamed_species(sandbox_path: str) -> ValidationResult:
    specimens_path = os.path.join(sandbox_path, "specimens.csv")
    taxonomy_path = os.path.join(sandbox_path, "active_taxonomy.json")
    exceptions_path = os.path.join(sandbox_path, "taxonomy_exceptions.json")
    output_path = os.path.join(sandbox_path, "specimen_labels.csv")

    for path in (specimens_path, taxonomy_path, exceptions_path, output_path):
        if not os.path.exists(path):
            return ValidationResult(False, details=f"Missing required file: {path}")

    specimens = _read_csv_rows(specimens_path)
    taxonomy = _read_json(taxonomy_path)
    exceptions = _read_json(exceptions_path)
    actual_rows = _read_csv_rows(output_path)

    err = _require_columns(specimens, {"specimen_id", "species_code", "region"}, "specimens.csv")
    if err:
        return err
    err = _require_columns(actual_rows, {"specimen_id", "species_label", "region"}, "specimen_labels.csv")
    if err:
        return err

    expected = []
    for row in specimens:
        sid = str(row.get("specimen_id", "")).strip()
        code = str(row.get("species_code", "")).strip()
        if sid in exceptions:
            label = str(exceptions[sid]).strip()
        else:
            label = str(taxonomy.get(code, "")).strip()
        if not label:
            return ValidationResult(False, details=f"No taxonomy label found for specimen_id={sid}, species_code={code}")
        expected.append((sid, label, str(row.get("region", "")).strip()))
    expected.sort(key=lambda x: x[0])

    actual = []
    for row in actual_rows:
        actual.append(
            (
                str(row.get("specimen_id", "")).strip(),
                str(row.get("species_label", "")).strip(),
                str(row.get("region", "")).strip(),
            )
        )
    actual.sort(key=lambda x: x[0])

    if actual != expected:
        return ValidationResult(False, details="specimen_labels.csv does not match expected taxonomy remapping.")
    return ValidationResult(True, score=1.0, details="specimen_labels.csv matches active taxonomy + exceptions.")


def validate_dream_court_transcript(sandbox_path: str) -> ValidationResult:
    testimony_path = os.path.join(sandbox_path, "testimony.csv")
    trust_path = os.path.join(sandbox_path, "active_source_trust.json")
    output_path = os.path.join(sandbox_path, "verdict_matrix.csv")

    for path in (testimony_path, trust_path, output_path):
        if not os.path.exists(path):
            return ValidationResult(False, details=f"Missing required file: {path}")

    rows = _read_csv_rows(testimony_path)
    trust = _read_json(trust_path)
    actual_rows = _read_csv_rows(output_path)

    err = _require_columns(rows, {"case_id", "citation_id", "source", "stance", "weight"}, "testimony.csv")
    if err:
        return err
    err = _require_columns(actual_rows, {"case_id", "verdict", "selected_citation"}, "verdict_matrix.csv")
    if err:
        return err

    by_case: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        stance = str(row.get("stance", "")).strip().upper()
        score = _to_float(row.get("weight")) * _to_float(trust.get(str(row.get("source", "")).strip(), 0.0))
        citation = str(row.get("citation_id", "")).strip()
        if case_id not in by_case:
            by_case[case_id] = {"support": 0.0, "oppose": 0.0, "support_c": [], "oppose_c": []}
        if stance == "SUPPORT":
            by_case[case_id]["support"] += score
            by_case[case_id]["support_c"].append((score, citation))
        else:
            by_case[case_id]["oppose"] += score
            by_case[case_id]["oppose_c"].append((score, citation))

    expected = []
    for case_id in sorted(by_case.keys()):
        item = by_case[case_id]
        if item["support"] >= item["oppose"]:
            verdict = "SUPPORT"
            pool = item["support_c"]
        else:
            verdict = "OPPOSE"
            pool = item["oppose_c"]
        if not pool:
            return ValidationResult(False, details=f"No winning citations available for case {case_id}")
        pool_sorted = sorted(pool, key=lambda x: (-x[0], x[1]))
        selected = pool_sorted[0][1]
        expected.append((case_id, verdict, selected))

    actual = []
    for row in actual_rows:
        actual.append(
            (
                str(row.get("case_id", "")).strip(),
                str(row.get("verdict", "")).strip().upper(),
                str(row.get("selected_citation", "")).strip(),
            )
        )
    actual.sort(key=lambda x: x[0])

    if actual != expected:
        return ValidationResult(False, details="verdict_matrix.csv mismatch against trust-weighted verdict computation.")
    return ValidationResult(True, score=1.0, details="verdict_matrix.csv matches trust-weighted verdicts.")


def _best_cargo_selection(items: list[dict[str, Any]], mass_limit_kg: float, volume_limit: float, forbidden_pairs: set[frozenset[str]]) -> tuple[list[str], float, float, float]:
    mandatory = [x for x in items if int(x["mandatory"]) == 1]
    optional = [x for x in items if int(x["mandatory"]) == 0]

    mandatory_ids = {x["item_id"] for x in mandatory}
    mandatory_mass = sum(_to_float(x["mass_kg"]) for x in mandatory)
    mandatory_vol = sum(_to_float(x["volume_m3"]) for x in mandatory)
    mandatory_value = sum(_to_float(x["value"]) for x in mandatory)

    best_key = None
    best_payload = None

    for r in range(len(optional) + 1):
        for combo in combinations(optional, r):
            selected = mandatory + list(combo)
            ids = sorted([str(x["item_id"]) for x in selected])
            id_set = set(ids)
            if any(pair.issubset(id_set) for pair in forbidden_pairs):
                continue

            total_mass = mandatory_mass + sum(_to_float(x["mass_kg"]) for x in combo)
            total_vol = mandatory_vol + sum(_to_float(x["volume_m3"]) for x in combo)
            total_value = mandatory_value + sum(_to_float(x["value"]) for x in combo)
            if total_mass > mass_limit_kg + 1e-9 or total_vol > volume_limit + 1e-9:
                continue

            # Prefer value, then lighter total mass, then deterministic lexicographic ids.
            key = (round(total_value, 6), -round(total_mass, 6), tuple(ids))
            if best_key is None or key > best_key:
                best_key = key
                best_payload = (ids, total_mass, total_vol, total_value)

    if best_payload is None:
        return sorted(mandatory_ids), mandatory_mass, mandatory_vol, mandatory_value
    return best_payload


def validate_lunar_cargo_ritual(sandbox_path: str) -> ValidationResult:
    items_path = os.path.join(sandbox_path, "cargo_items.csv")
    constraints_path = os.path.join(sandbox_path, "active_constraints.json")
    unit_profile_path = os.path.join(sandbox_path, "active_unit_profile.json")
    output_path = os.path.join(sandbox_path, "cargo_plan.json")

    for path in (items_path, constraints_path, unit_profile_path, output_path):
        if not os.path.exists(path):
            return ValidationResult(False, details=f"Missing required file: {path}")

    items = _read_csv_rows(items_path)
    constraints = _read_json(constraints_path)
    unit_profile = _read_json(unit_profile_path)
    actual = _read_json(output_path)

    err = _require_columns(items, {"item_id", "mass_kg", "volume_m3", "value", "mandatory"}, "cargo_items.csv")
    if err:
        return err
    if not isinstance(actual, dict):
        return ValidationResult(False, details="cargo_plan.json must contain a JSON object.")

    mass_unit = str(unit_profile.get("mass_unit", "kg")).strip().lower()
    if "max_mass_kg" in constraints:
        mass_limit_kg = _to_float(constraints["max_mass_kg"])
    elif "max_mass_lb" in constraints:
        mass_limit_kg = _to_float(constraints["max_mass_lb"]) / 2.20462
    else:
        return ValidationResult(False, details="active_constraints.json missing max mass key.")
    volume_limit = _to_float(constraints.get("max_volume_m3"))
    forbidden = constraints.get("forbidden_pairs", [])
    forbidden_pairs = {frozenset((str(p[0]), str(p[1]))) for p in forbidden if isinstance(p, list) and len(p) == 2}

    expected_ids, expected_mass, expected_vol, expected_value = _best_cargo_selection(
        items=items,
        mass_limit_kg=mass_limit_kg,
        volume_limit=volume_limit,
        forbidden_pairs=forbidden_pairs,
    )

    selected = actual.get("selected_items")
    if not isinstance(selected, list):
        return ValidationResult(False, details="cargo_plan.json must contain selected_items as a list.")
    actual_ids = sorted(str(x) for x in selected)
    if actual_ids != expected_ids:
        return ValidationResult(False, details=f"selected_items mismatch. Expected {expected_ids}, got {actual_ids}.")

    actual_mass = _to_float(actual.get("total_mass_kg"))
    actual_vol = _to_float(actual.get("total_volume_m3"))
    actual_value = _to_float(actual.get("total_value"))

    if abs(actual_mass - expected_mass) > 1e-6:
        return ValidationResult(False, details=f"total_mass_kg mismatch. Expected {expected_mass:.6f}, got {actual_mass:.6f}.")
    if abs(actual_vol - expected_vol) > 1e-6:
        return ValidationResult(False, details=f"total_volume_m3 mismatch. Expected {expected_vol:.6f}, got {actual_vol:.6f}.")
    if abs(actual_value - expected_value) > 1e-6:
        return ValidationResult(False, details=f"total_value mismatch. Expected {expected_value:.6f}, got {actual_value:.6f}.")

    return ValidationResult(True, score=1.0, details=f"cargo_plan.json matches optimal feasible plan under active {mass_unit} constraints.")


def validate_paradox_lab_protocol(sandbox_path: str) -> ValidationResult:
    steps_path = os.path.join(sandbox_path, "protocol_steps.csv")
    subs_path = os.path.join(sandbox_path, "substitutions.json")
    status_path = os.path.join(sandbox_path, "active_lab_status.json")
    plan_path = os.path.join(sandbox_path, "protocol_plan.csv")
    summary_path = os.path.join(sandbox_path, "run_summary.json")

    for path in (steps_path, subs_path, status_path, plan_path, summary_path):
        if not os.path.exists(path):
            return ValidationResult(False, details=f"Missing required file: {path}")

    steps = _read_csv_rows(steps_path)
    subs = _read_json(subs_path)
    status = _read_json(status_path)
    plan_rows = _read_csv_rows(plan_path)
    summary = _read_json(summary_path)

    err = _require_columns(steps, {"step_id", "instrument", "duration_min", "yield_points", "required"}, "protocol_steps.csv")
    if err:
        return err
    err = _require_columns(plan_rows, {"step_id", "instrument", "duration_min"}, "protocol_plan.csv")
    if err:
        return err
    if not isinstance(summary, dict):
        return ValidationResult(False, details="run_summary.json must be a JSON object.")

    removed = {str(x).strip() for x in status.get("removed_instruments", [])}
    expected = []
    total_duration = 0
    total_yield = 0
    for row in steps:
        required = _to_int(row.get("required"))
        duration = _to_int(row.get("duration_min"))
        yield_points = _to_int(row.get("yield_points"))
        include = required == 1 or duration <= 5
        if not include:
            continue

        instrument = str(row.get("instrument", "")).strip()
        if instrument in removed:
            sub = subs.get(instrument) or {}
            instrument = str(sub.get("replacement", instrument)).strip()
            duration += _to_int(sub.get("penalty_duration"))

        expected.append((str(row.get("step_id", "")).strip(), instrument, duration))
        total_duration += duration
        total_yield += yield_points

    actual = []
    for row in plan_rows:
        actual.append(
            (
                str(row.get("step_id", "")).strip(),
                str(row.get("instrument", "")).strip(),
                _to_int(row.get("duration_min")),
            )
        )

    if actual != expected:
        return ValidationResult(False, details=f"protocol_plan.csv mismatch. Expected {expected}, got {actual}.")

    if _to_int(summary.get("total_duration_min")) != total_duration:
        return ValidationResult(False, details=f"run_summary total_duration_min mismatch. Expected {total_duration}.")
    if _to_int(summary.get("total_yield_points")) != total_yield:
        return ValidationResult(False, details=f"run_summary total_yield_points mismatch. Expected {total_yield}.")

    return ValidationResult(True, score=1.0, details="Protocol plan and summary match active lab constraints.")

