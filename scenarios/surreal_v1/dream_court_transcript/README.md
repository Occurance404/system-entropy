# Dream Court Transcript

## Goal
Infer final verdicts from conflicting transcript segments and produce verdict_matrix.csv.

## Shock
One witness source is retroactively marked unreliable; recompute verdicts using updated trust rules.

## Validator Contract
verdict_matrix.csv must contain exact verdicts and citation IDs per case.

## Data Authoring Checklist

- Place all agent-visible files under `inputs/`.
- Place validator-only reference files under `expected/`.
- Keep perturbation language concrete and testable.
- Keep validator deterministic and side-effect free.
