# Archive of Impossible Dates

## Goal
Reconstruct a normalized historical timeline from mixed calendar records and produce timeline.csv.

## Shock
Calendar standard has changed; convert all prior records into the new canonical calendar before finalizing timeline.csv.

## Validator Contract
timeline.csv must have exact normalized dates, stable ordering, and required columns.

## Data Authoring Checklist

- Place all agent-visible files under `inputs/`.
- Place validator-only reference files under `expected/`.
- Keep perturbation language concrete and testable.
- Keep validator deterministic and side-effect free.
