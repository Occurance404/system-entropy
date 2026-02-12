# City of Duplicate Identities

## Goal
Resolve entities across messy records and output master_entities.csv.

## Shock
Primary keys are no longer unique; identity must be composite and conflict-aware.

## Validator Contract
master_entities.csv must exactly match expected deduped entities and links.

## Data Authoring Checklist

- Place all agent-visible files under `inputs/`.
- Place validator-only reference files under `expected/`.
- Keep perturbation language concrete and testable.
- Keep validator deterministic and side-effect free.
