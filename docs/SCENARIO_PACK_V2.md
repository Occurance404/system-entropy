# Scenario Pack V2 (Hard, Publishable, Runnable)

This pack is designed for your current harness (`setup_ops` + `definitions` + `perturbation_ops` + `validation_ops`) and avoids tasks solvable in a few shots.

## Design Rules
- Multi-phase outputs (at least 3 files per run)
- Cross-file consistency checks (not single-file completion)
- Deterministic validators (exact counts/invariants, not subjective grading)
- Mid-run non-stationarity (schema/policy/resource shocks)
- Idempotency checks (rerun-safe behavior required)

## Recommended Pack (implement these 4 families)

### 1) `incident_reconstruction_baseline` / `incident_reconstruction_shock`
- Core problem: reconstruct a production incident from `gateway.log`, `worker.log`, `deploy_events.json`, and `trace_spans.json`.
- Required outputs:
  - `incident_timeline.csv` (ordered event chain)
  - `root_cause.md` (must name exact failing component + trigger commit)
  - `impacted_requests.txt` (unique request IDs)
- Shock (step 6):
  - Add `late_arrival.log` with delayed events and 2-minute clock skew.
  - New instruction: timeline must be corrected using `trace_id`, not raw timestamps.
- Validator:
  - Correct request count, correct root cause key phrase, exact timeline ordering by canonical trace sequence.
- Why hard:
  - Multi-source correlation + contradictory time signals.

### 2) `migration_idempotency_baseline` / `migration_idempotency_shock`
- Core problem: migrate mixed legacy data into SQLite with strict invariants.
- Setup files:
  - `legacy/users_2019.csv` (UTF-8), `legacy/users_2020.csv` (latin1), `legacy/orders.json`.
  - `target_schema.sql`.
- Required outputs:
  - `migrate.py`
  - `production.db`
  - `migration_report.json` (counts, rejects, duplicate reasons)
- Shock (step 7):
  - New file `legacy/users_2018.pipe`.
  - Constraint change: migration must be idempotent; running `migrate.py` twice cannot change row counts.
- Validator:
  - Row totals per table, reject counts, normalized dates, and explicit rerun check (run script twice).
- Why hard:
  - Encoding + format heterogeneity + idempotency under changing inputs.

### 3) `policy_redaction_baseline` / `policy_redaction_shock`
- Core problem: transform support transcripts into safe release files under privacy policy.
- Setup files:
  - `tickets/*.txt`
  - `policy_v1.md`
  - `allowed_entities.csv`
- Required outputs:
  - `redacted/*.txt`
  - `entity_map.csv`
  - `compliance_report.md`
- Shock (step 5):
  - `policy_v2.md` replaces v1: names/emails must be pseudonymized consistently across files, but order IDs must remain unchanged.
- Validator:
  - Zero leaked PII patterns, stable pseudonyms for same identity, preserved order IDs, report sections present.
- Why hard:
  - Competing constraints (privacy vs referential consistency).

### 4) `resource_scheduler_baseline` / `resource_scheduler_shock`
- Core problem: schedule jobs with machine constraints and deadlines.
- Setup files:
  - `jobs.csv`, `machines.csv`, `changeover_costs.csv`.
- Required outputs:
  - `schedule.csv`
  - `objective_breakdown.json` (lateness, changeover, utilization)
  - `scheduler.py`
- Shock (step 8):
  - Machine outage event: one machine unavailable for a fixed window.
  - New rule: already-started jobs cannot be rescheduled.
- Validator:
  - No overlapping assignments, all hard constraints satisfied, objective below threshold, outage respected.
- Why hard:
  - Constraint reasoning and replanning under irreversible commitments.

## Optional Stretch Family

### 5) `api_contract_drift_baseline` / `api_contract_drift_shock`
- Core problem: integrate 3 mocked APIs and produce reconciled `settlements.csv`.
- Shock:
  - Payment response contract changes + rate-limit 429 + idempotency key required.
- Validator:
  - Correct retries/backoff markers in logs, no duplicate charges, complete settlements.
- Use this only if time remains after the 4 core families.

## Anti-"Few Shots" Guardrails
- Minimum output set per scenario: 3 artifacts.
- Validator must check at least 4 invariants.
- Add hidden edge rows in setup data (missing fields, duplicate IDs, malformed dates).
- Shock instruction must force code/path rewrite, not parameter tweak.

## 3-Month Practical Scope
- Build only 4 families above (8 scenarios total baseline+shock).
- Repeats target: 8-12 per model, 3 models.
- Success criterion for paper: shock causes significant drop in validator pass rate and/or increase in steps/tokens.

## Implementation Order
1. `migration_idempotency_*` (highest paper value, deterministic)
2. `incident_reconstruction_*` (strong non-stationarity signal)
3. `resource_scheduler_*` (hard planning benchmark)
4. `policy_redaction_*` (compliance-style generalization)

## Harness Mapping Checklist
- Add scenarios in `src/scenarios/definitions.py`:
  - 8 IDs (4 baseline + 4 shock), perturbation steps at 5-8.
- Add setup generators in `src/scenarios/setup_ops.py`:
  - one setup function per family, map both baseline/shock IDs to same setup.
- Add perturbation mutators in `src/scenarios/perturbation_ops.py`:
  - one deterministic mutator per shock scenario.
- Add validators in `src/scenarios/validation_ops.py`:
  - baseline validator and shock validator for each family.
- Add suite file `benchmarks/suite_paper_v2.json`:
  - include all 8 scenarios with `max_steps` around 45-80.
- Dry run:
  - run each scenario once with local model; ensure validator pass/fail is deterministic.
