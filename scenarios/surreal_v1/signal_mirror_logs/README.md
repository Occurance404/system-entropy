# Signal in the Mirror Logs

## Goal
Detect anomaly chains from mirrored logs and output incidents.json.

## Shock
Mirror feed shifts to delayed and out-of-order replay; chain reconstruction must remain correct.

## Validator Contract
incidents.json must contain exact chain ordering and root-cause IDs.

## Data Authoring Checklist

- Place all agent-visible files under `inputs/`.
- Place validator-only reference files under `expected/`.
- Keep perturbation language concrete and testable.
- Keep validator deterministic and side-effect free.
