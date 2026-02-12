# The Missing Axiom

## Goal
Complete a symbolic transformation pipeline and output transforms.json.

## Shock
One axiom is invalidated; recompute a consistent rule set and update transforms.

## Validator Contract
transforms.json must satisfy hidden consistency checks and expected outputs.

## Data Authoring Checklist

- Place all agent-visible files under `inputs/`.
- Place validator-only reference files under `expected/`.
- Keep perturbation language concrete and testable.
- Keep validator deterministic and side-effect free.
