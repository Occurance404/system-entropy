# Lunar Cargo Ritual

## Goal
Build a feasible cargo plan under mass/volume/compatibility constraints and output cargo_plan.json.

## Shock
Unit system flips mid-run and a new incompatibility pair is introduced.

## Validator Contract
cargo_plan.json must satisfy all constraints and pass deterministic feasibility checks.

## Data Authoring Checklist

- Place all agent-visible files under `inputs/`.
- Place validator-only reference files under `expected/`.
- Keep perturbation language concrete and testable.
- Keep validator deterministic and side-effect free.
