# Oracle Contract Amendment

## Goal
Implement rule evaluation from contract clauses and output rule_engine.py plus decisions.csv.

## Shock
A superseding amendment applies retroactively with higher precedence than earlier clauses.

## Validator Contract
Hidden/public rule tests must pass on decisions.csv and rule behavior.

## Data Authoring Checklist

- Place all agent-visible files under `inputs/`.
- Place validator-only reference files under `expected/`.
- Keep perturbation language concrete and testable.
- Keep validator deterministic and side-effect free.
