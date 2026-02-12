from __future__ import annotations

import os
from typing import Any

from .common import (
    ValidationResult,
    _read_csv_rows,
    _read_json,
    _require_columns,
    _to_float,
    _to_int,
)

def validate_oracle_contract_amendment(sandbox_path: str) -> ValidationResult:
    contracts_path = os.path.join(sandbox_path, "contracts.csv")
    rules_path = os.path.join(sandbox_path, "active_rules.json")
    output_path = os.path.join(sandbox_path, "decisions.csv")

    for path in (contracts_path, rules_path, output_path):
        if not os.path.exists(path):
            return ValidationResult(False, details=f"Missing required file: {path}")

    contracts = _read_csv_rows(contracts_path)
    rules = _read_json(rules_path)
    actual_rows = _read_csv_rows(output_path)

    err = _require_columns(contracts, {"contract_id", "base_amount", "usage_units", "customer_tier", "late_days"}, "contracts.csv")
    if err:
        return err
    err = _require_columns(actual_rows, {"contract_id", "amount_due", "applied_rate", "discount", "late_fee"}, "decisions.csv")
    if err:
        return err

    rate = _to_float(rules.get("rate_per_unit"))
    retro = _to_float(rules.get("retroactive_multiplier"), 1.0)
    late_fee_rule = _to_float(rules.get("late_fee"))
    discounts = rules.get("tier_discounts") or {}

    expected = {}
    for row in contracts:
        cid = str(row.get("contract_id", "")).strip()
        base_amount = _to_float(row.get("base_amount"))
        usage = _to_float(row.get("usage_units"))
        tier = str(row.get("customer_tier", "")).strip()
        late_days = _to_int(row.get("late_days"))
        discount_pct = _to_float(discounts.get(tier), 0.0)

        gross = (base_amount + usage * rate) * retro
        discount_value = gross * discount_pct
        late_fee = late_fee_rule if late_days > 0 else 0.0
        amount_due = round(gross - discount_value + late_fee, 2)
        expected[cid] = (
            amount_due,
            round(rate, 6),
            round(discount_value, 2),
            round(late_fee, 2),
        )

    for row in actual_rows:
        cid = str(row.get("contract_id", "")).strip()
        if cid not in expected:
            return ValidationResult(False, details=f"Unexpected contract_id in decisions.csv: {cid}")
        exp_amount, exp_rate, exp_discount, exp_late = expected[cid]
        act_amount = round(_to_float(row.get("amount_due")), 2)
        act_rate = round(_to_float(row.get("applied_rate")), 6)
        act_discount = round(_to_float(row.get("discount")), 2)
        act_late = round(_to_float(row.get("late_fee")), 2)
        if (act_amount, act_rate, act_discount, act_late) != (exp_amount, exp_rate, exp_discount, exp_late):
            return ValidationResult(
                False,
                details=(
                    f"Mismatch for {cid}. Expected {(exp_amount, exp_rate, exp_discount, exp_late)}, "
                    f"got {(act_amount, act_rate, act_discount, act_late)}."
                ),
            )

    if len(actual_rows) != len(expected):
        return ValidationResult(False, details=f"decisions.csv row count mismatch. Expected {len(expected)}, got {len(actual_rows)}.")

    return ValidationResult(True, score=1.0, details="decisions.csv matches active contract rules.")


def validate_city_duplicate_identities(sandbox_path: str) -> ValidationResult:
    records_path = os.path.join(sandbox_path, "citizen_records.csv")
    policy_path = os.path.join(sandbox_path, "active_identity_policy.json")
    output_path = os.path.join(sandbox_path, "master_entities.csv")

    for path in (records_path, policy_path, output_path):
        if not os.path.exists(path):
            return ValidationResult(False, details=f"Missing required file: {path}")

    records = _read_csv_rows(records_path)
    policy = _read_json(policy_path)
    actual_rows = _read_csv_rows(output_path)
    keys = [str(k).strip() for k in policy.get("keys", []) if str(k).strip()]
    if not keys:
        return ValidationResult(False, details="active_identity_policy.json contains no keys.")

    err = _require_columns(records, {"record_id", "full_name", "dob", "email", "city"}, "citizen_records.csv")
    if err:
        return err
    err = _require_columns(actual_rows, {"entity_id", "record_ids", "canonical_name", "count"}, "master_entities.csv")
    if err:
        return err

    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in records:
        key = tuple(str(row.get(k, "")).strip().lower() for k in keys)
        groups.setdefault(key, []).append(row)

    sorted_groups = sorted(groups.values(), key=lambda g: min(_to_int(r.get("record_id")) for r in g))
    expected = []
    for idx, group in enumerate(sorted_groups, start=1):
        ids_sorted = sorted((_to_int(r.get("record_id")) for r in group))
        record_ids = ";".join(str(x) for x in ids_sorted)
        canonical_name = sorted(str(r.get("full_name", "")).strip() for r in group)[0]
        expected.append((f"E{idx:03d}", record_ids, canonical_name, len(group)))

    actual = []
    for row in actual_rows:
        actual.append(
            (
                str(row.get("entity_id", "")).strip(),
                str(row.get("record_ids", "")).strip(),
                str(row.get("canonical_name", "")).strip(),
                _to_int(row.get("count")),
            )
        )
    actual.sort(key=lambda x: x[0])
    expected.sort(key=lambda x: x[0])

    if actual != expected:
        return ValidationResult(False, details=f"master_entities.csv mismatch. Expected {expected}, got {actual}.")
    return ValidationResult(True, score=1.0, details="master_entities.csv matches active identity policy clustering.")


def validate_signal_mirror_logs(sandbox_path: str) -> ValidationResult:
    primary_path = os.path.join(sandbox_path, "primary_log.csv")
    mirror_path = os.path.join(sandbox_path, "mirror_log.csv")
    policy_path = os.path.join(sandbox_path, "active_ordering_policy.json")
    output_path = os.path.join(sandbox_path, "incidents.json")

    for path in (primary_path, mirror_path, policy_path, output_path):
        if not os.path.exists(path):
            return ValidationResult(False, details=f"Missing required file: {path}")

    primary_rows = _read_csv_rows(primary_path)
    mirror_rows = _read_csv_rows(mirror_path)
    policy = _read_json(policy_path)
    actual = _read_json(output_path)

    required = {"event_id", "incident_id", "seq", "timestamp", "event_type", "severity"}
    err = _require_columns(primary_rows, required, "primary_log.csv")
    if err:
        return err
    err = _require_columns(mirror_rows, required, "mirror_log.csv")
    if err:
        return err

    sort_key = str(policy.get("sort_key", "timestamp")).strip()
    if sort_key not in ("timestamp", "seq"):
        return ValidationResult(False, details=f"Unsupported sort_key in active_ordering_policy.json: {sort_key}")

    combined = {}
    for row in primary_rows + mirror_rows:
        event_id = str(row.get("event_id", "")).strip()
        if not event_id:
            continue
        prior = combined.get(event_id)
        if prior is None:
            combined[event_id] = dict(row)
            continue
        # Keep deterministic best version of duplicate events.
        if _to_int(row.get("timestamp")) < _to_int(prior.get("timestamp")):
            combined[event_id] = dict(row)

    by_incident: dict[str, list[dict[str, str]]] = {}
    for row in combined.values():
        incident_id = str(row.get("incident_id", "")).strip()
        by_incident.setdefault(incident_id, []).append(row)

    expected = []
    for incident_id in sorted(by_incident.keys()):
        rows = sorted(
            by_incident[incident_id],
            key=lambda r: (_to_int(r.get(sort_key)), str(r.get("event_id", ""))),
        )
        chain = [str(r.get("event_id", "")).strip() for r in rows]
        precursor = next((str(r.get("event_id", "")).strip() for r in rows if str(r.get("event_type", "")).strip() == "PRECURSOR"), "")
        root_event = precursor or (chain[0] if chain else "")
        expected.append(
            {
                "incident_id": incident_id,
                "chain": chain,
                "event_count": len(chain),
                "root_event": root_event,
            }
        )

    if isinstance(actual, dict):
        actual_list = actual.get("incidents")
    else:
        actual_list = actual
    if not isinstance(actual_list, list):
        return ValidationResult(False, details="incidents.json must be a list or {'incidents': [...]} object.")

    normalized_actual = []
    for row in actual_list:
        if not isinstance(row, dict):
            return ValidationResult(False, details="incidents.json contains non-object entries.")
        chain = row.get("chain")
        if not isinstance(chain, list):
            return ValidationResult(False, details="Each incident must include chain as a list.")
        normalized_actual.append(
            {
                "incident_id": str(row.get("incident_id", "")).strip(),
                "chain": [str(x).strip() for x in chain],
                "event_count": _to_int(row.get("event_count")),
                "root_event": str(row.get("root_event", "")).strip(),
            }
        )
    normalized_actual.sort(key=lambda x: x["incident_id"])

    if normalized_actual != expected:
        return ValidationResult(False, details=f"incidents.json mismatch. Expected {expected}, got {normalized_actual}.")
    return ValidationResult(True, score=1.0, details="incidents.json matches reconstructed incident chains.")


def validate_missing_axiom(sandbox_path: str) -> ValidationResult:
    expressions_path = os.path.join(sandbox_path, "expressions.csv")
    axioms_path = os.path.join(sandbox_path, "active_axioms.json")
    output_path = os.path.join(sandbox_path, "transforms.json")

    for path in (expressions_path, axioms_path, output_path):
        if not os.path.exists(path):
            return ValidationResult(False, details=f"Missing required file: {path}")

    expressions = _read_csv_rows(expressions_path)
    axioms = _read_json(axioms_path)
    actual = _read_json(output_path)

    err = _require_columns(expressions, {"expr_id", "x", "y", "formula_id"}, "expressions.csv")
    if err:
        return err

    expected = []
    for row in expressions:
        expr_id = str(row.get("expr_id", "")).strip()
        formula_id = str(row.get("formula_id", "")).strip()
        if formula_id not in axioms:
            return ValidationResult(False, details=f"Formula {formula_id} missing in active_axioms.json.")
        coeff = axioms[formula_id]
        a = _to_float(coeff.get("a"))
        b = _to_float(coeff.get("b"))
        c = _to_float(coeff.get("c"))
        x = _to_float(row.get("x"))
        y = _to_float(row.get("y"))
        expected.append({"expr_id": expr_id, "value": a * x + b * y + c})
    expected.sort(key=lambda x: x["expr_id"])

    if isinstance(actual, dict):
        actual_list = actual.get("results")
    else:
        actual_list = actual
    if not isinstance(actual_list, list):
        return ValidationResult(False, details="transforms.json must be a list or {'results': [...]} object.")

    normalized_actual = []
    for row in actual_list:
        if not isinstance(row, dict):
            return ValidationResult(False, details="transforms.json contains non-object entries.")
        normalized_actual.append({"expr_id": str(row.get("expr_id", "")).strip(), "value": _to_float(row.get("value"))})
    normalized_actual.sort(key=lambda x: x["expr_id"])

    if len(normalized_actual) != len(expected):
        return ValidationResult(False, details=f"transforms.json row count mismatch. Expected {len(expected)}, got {len(normalized_actual)}.")
    for exp, act in zip(expected, normalized_actual):
        if exp["expr_id"] != act["expr_id"] or abs(exp["value"] - act["value"]) > 1e-6:
            return ValidationResult(False, details=f"Mismatch for expr_id={exp['expr_id']}. Expected {exp['value']}, got {act['value']}.")

    return ValidationResult(True, score=1.0, details="transforms.json matches active axioms.")


def validate_bureau_contradictory_forms(sandbox_path: str) -> ValidationResult:
    forms_path = os.path.join(sandbox_path, "forms.csv")
    policy_path = os.path.join(sandbox_path, "active_policy.json")
    output_path = os.path.join(sandbox_path, "canonical_records.csv")

    for path in (forms_path, policy_path, output_path):
        if not os.path.exists(path):
            return ValidationResult(False, details=f"Missing required file: {path}")

    forms = _read_csv_rows(forms_path)
    policy = _read_json(policy_path)
    actual_rows = _read_csv_rows(output_path)

    err = _require_columns(forms, {"record_id", "name", "tax_id_gov", "tax_id_vendor", "address_gov", "address_vendor"}, "forms.csv")
    if err:
        return err
    err = _require_columns(actual_rows, {"record_id", "name", "tax_id", "address", "source_policy"}, "canonical_records.csv")
    if err:
        return err

    tax_pref = str(policy.get("tax_precedence", "vendor")).strip().lower()
    addr_pref = str(policy.get("address_precedence", "vendor")).strip().lower()
    if tax_pref not in ("gov", "vendor") or addr_pref not in ("gov", "vendor"):
        return ValidationResult(False, details="active_policy.json contains invalid precedence values.")

    expected = []
    for row in forms:
        record_id = str(row.get("record_id", "")).strip()
        tax_id = str(row.get("tax_id_gov") if tax_pref == "gov" else row.get("tax_id_vendor")).strip()
        address = str(row.get("address_gov") if addr_pref == "gov" else row.get("address_vendor")).strip()
        expected.append(
            (
                record_id,
                str(row.get("name", "")).strip(),
                tax_id,
                address,
                f"{tax_pref}/{addr_pref}",
            )
        )
    expected.sort(key=lambda x: x[0])

    actual = []
    for row in actual_rows:
        actual.append(
            (
                str(row.get("record_id", "")).strip(),
                str(row.get("name", "")).strip(),
                str(row.get("tax_id", "")).strip(),
                str(row.get("address", "")).strip(),
                str(row.get("source_policy", "")).strip().lower(),
            )
        )
    actual.sort(key=lambda x: x[0])

    if actual != expected:
        return ValidationResult(False, details=f"canonical_records.csv mismatch. Expected {expected}, got {actual}.")

    return ValidationResult(True, score=1.0, details="canonical_records.csv matches active authority precedence policy.")

