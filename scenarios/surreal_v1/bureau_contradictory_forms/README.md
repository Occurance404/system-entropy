# Bureau of Contradictory Forms

## Goal
Normalize contradictory forms into canonical_records.csv under policy constraints.

## Shock
Policy update changes authority precedence; recompute normalized records accordingly.

## Validator Contract
canonical_records.csv must exactly match schema and expected row-level outputs.

## Data Authoring Checklist

- Place all agent-visible files under `inputs/`.
- Place validator-only reference files under `expected/`.
- Keep perturbation language concrete and testable.
- Keep validator deterministic and side-effect free.
