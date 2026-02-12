# Museum of Renamed Species

## Goal
Classify specimen records using taxonomy_map.json and output specimen_labels.csv.

## Shock
Taxonomy patch deprecates old labels and introduces exception overrides for specific specimen IDs.

## Validator Contract
specimen_labels.csv must match expected remapping and exception handling exactly.

## Data Authoring Checklist

- Place all agent-visible files under `inputs/`.
- Place validator-only reference files under `expected/`.
- Keep perturbation language concrete and testable.
- Keep validator deterministic and side-effect free.
