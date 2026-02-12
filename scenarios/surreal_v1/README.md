# Surreal Scenario Pack v1

This pack is a file-first scaffold for high-reasoning experiments.
It is intentionally agent-agnostic: no scenario-specific logic in the agent.

## Contract Per Scenario

Each scenario directory contains:

- `prompt.md`: task shown to the agent.
- `inputs/`: files exposed to the agent workspace.
- `perturbations.json`: step-triggered requirement changes.
- `validator.py`: deterministic validator returning `{passed, score, details}`.
- `expected/`: hidden references used by validator only.
- `manifest.json`: metadata + hashes/version placeholders.
- `README.md`: scenario intent, risks, and success criteria.

## Notes

- Keep all source artifacts in JSON/CSV/JSONL alongside SQLite ingestion.
- Avoid changing validator rules mid-experiment batch.
- Version bump `manifest.json` when inputs or perturbations change.
