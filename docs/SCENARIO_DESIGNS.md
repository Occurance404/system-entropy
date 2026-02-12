# Realistic Stress-Test Scenario Designs

These scenarios are designed to **actually stress LLM agents** the way real-world tasks do:
- Force 15K+ token context accumulation
- Create genuine ambiguity that causes SCR spikes
- Include realistic perturbations (not arbitrary filter changes)
- Have clear success/failure validation

---

## Scenario 1: Microservices Debugging Nightmare

**Why it works:** Forces reading 4+ files, understanding call chains, dealing with ambiguous error messages.

### Initial Prompt
```
TASK: There's a bug in our microservices system. Users report "Order failed" 
but no error logs appear. 

The system has:
- gateway/app.py (Flask, routes to services)
- orders/service.py (handles order creation)
- inventory/service.py (checks stock)
- notifications/service.py (sends emails)
- shared/models.py (SQLAlchemy models)
- docker-compose.yml (service config)
- logs/ (empty — that's the bug)

Find the bug, fix it, and write a test that would have caught it.

CONSTRAINT: The bug is a SILENT FAILURE (something swallows an exception). 
You must trace the full call path.
```

### Perturbation (Step 6)
```
URGENT: The staging environment just crashed. The bug is now in production 
affecting real customers. Also, we realized that `notifications/service.py` 
was deployed from a different branch and has a DIFFERENT version of the 
API contract. Fix both issues.
```

### What Makes It Hard
- Multiple files must be read to understand call flow
- Silent failures are hard to trace without logs
- The perturbation adds a second, unrelated bug mid-task
- Agent must maintain mental model of 4 services

### Validation
- [ ] Bug identified correctly (exception swallowed in X file)
- [ ] Fix applied
- [ ] Test written that reproduces the original bug

---

## Scenario 2: Data Migration Hell (High Token Load)

**Why it works:** Forces 15K+ token context from reading messy data, dealing with encoding issues.

### Initial Prompt
```
TASK: Migrate data from legacy system to new schema.

Files:
- legacy/customers_2019.csv (50000 rows, Windows-1252, dates as DD/MM/YYYY)
- legacy/customers_2020.csv (50000 rows, UTF-8, dates as YYYY-MM-DD)  
- legacy/orders_combined.json (nested, some objects have 'customerID', 
  others have 'customer_id')
- legacy/README.txt (describes fields but is OUTDATED and WRONG about 
  3 column names)
- new_schema/schema.sql (the target)

Requirements:
1. All data must be migrated. Zero data loss.
2. Names must be normalized (proper case).
3. Phone numbers must be E.164 format.
4. Any row that fails validation goes to `rejected_rows.csv` with reason.
5. Write `migration_report.md` with row counts and validation stats.
```

### Perturbation (Step 8)
```
STOP. We just found `legacy/customers_2018.csv` on an old backup drive. 
It uses pipe-delimited format and has 12 additional columns not in your 
current schema. You must also migrate this file. The deadline is unchanged.
```

### What Makes It Hard
- Inconsistent date formats across files
- Encoding issues cause UnicodeDecodeError traps
- The README is intentionally wrong (tests hallucination detection)
- Perturbation adds a completely different file format

### Validation
- [ ] All rows from all 3 CSVs present in final DB
- [ ] rejected_rows.csv exists with reasons
- [ ] migration_report.md has accurate counts

---

## Scenario 3: API Integration Cascade

**Why it works:** Tests multi-system thinking, error handling, and dealing with external failures.

### Initial Prompt
```
TASK: Build an order processing pipeline that:
1. Reads orders from `incoming_orders.json`
2. Validates inventory by calling [mock] Inventory API
3. Calculates shipping via [mock] Shipping API  
4. Charges payment via [mock] Payment API
5. Writes results to `processed_orders.csv`

Mock APIs are defined in `api_mocks.py`. You can call them.

CRITICAL: If ANY API fails, the order must be ROLLED BACK completely. 
No partial processing.

You must handle: timeouts, 4xx errors, 5xx errors, malformed responses.

Write `order_processor.py` and `test_order_processor.py`.
```

### Perturbation (Step 5)
```
The Payment API vendor just changed their response format:
- `status` field is now `payment_status`
- `success` value is now `APPROVED`
- They now require an `idempotency_key` header or reject duplicate charges

Update your integration.
```

### Perturbation (Step 9)
```
Load test failed. When processing 100 concurrent orders, the Inventory API 
starts returning 429 (Rate Limited). Implement exponential backoff and 
retry logic. Do not lose any orders.
```

### What Makes It Hard
- Rollback logic is notoriously tricky
- Error handling for 4 different failure modes
- Two perturbations that change API behavior
- Concurrency issues in second perturbation

### Validation
- [ ] processed_orders.csv has correct data
- [ ] No partial orders (rollback works)
- [ ] Retry logic handles 429s

---

## Scenario 4: Merge Conflict Resolution

**Why it works:** This is what coding agents ACTUALLY fail at in real repos.

### Initial Prompt
```
TASK: You are resolving a merge conflict in a codebase.

Two branches modified the same files:
- feature-auth modified: auth.py, models.py, tests/test_auth.py
- feature-billing modified: billing.py, models.py, tests/test_billing.py

Both branches changed `models.py` in INCOMPATIBLE ways:
- feature-auth added `User.mfa_enabled: bool`
- feature-billing renamed `User.email` to `User.primary_email`

The merge has conflicts. Files are in `workspace/`:
- models.py (has <<< HEAD === >>> markers)
- auth.py (references old User model)
- billing.py (references renamed field)
- tests/test_auth.py
- tests/test_billing.py

TASK: Resolve ALL conflicts such that BOTH features work. All tests must pass.
```

### Perturbation (Step 4)
```
Actually, we need both `email` AND `primary_email` to exist during migration. 
Add a `@property` that makes `email` an alias for `primary_email` and log a 
deprecation warning. Update all usages accordingly.
```

### What Makes It Hard
- Merge conflicts are cognitively complex
- Both branches have valid changes that must coexist
- Tests from both branches must pass
- Perturbation adds backwards compatibility requirement

### Validation
- [ ] models.py has no conflict markers
- [ ] auth.py imports/uses correct model
- [ ] billing.py uses renamed field
- [ ] All tests pass

---

## Scenario 5: Debugging Without Reproduction

**Why it works:** Forces the agent to reason from incomplete information — a classic SCR trap.

### Initial Prompt
```
TASK: A production bug was reported. We don't have reproduction steps.

Customer report (in `bug_report.txt`):
"After I click save, sometimes it shows a spinner forever and nothing happens. 
Happens maybe 1 in 5 times. Only started last week."

You have:
- frontend/app.js (React, 800 lines)
- backend/api.py (Flask, handles /save endpoint)
- backend/tasks.py (Celery async tasks)
- logs/backend.log (last 24 hours, DOES NOT have the bug - it's intermittent)
- git log (shows what changed last week)

TASK: 
1. Analyze the code and git diff to hypothesize what causes the intermittent bug.
2. Write a fix.
3. Write a test that would reproduce the race condition.
```

### Perturbation (Step 5)
```
The customer just sent more info: 
"It only happens when I save really fast, like clicking twice."

This is a double-submit race condition. Also, we just realized the Celery 
worker was restarted last week with `--concurrency=1` instead of `4`. 
That might be related.

Update your analysis.
```

### What Makes It Hard
- Bug is not in logs (intermittent)
- Must reason from code + git history
- Race conditions are subtle
- Perturbation changes the hypothesis mid-investigation

### Validation
- [ ] Correct root cause identified
- [ ] Fix prevents double-submit
- [ ] Test reproduces race condition

---

## Implementation Priority

For your paper, I recommend implementing in this order:

1. **Scenario 2 (Data Migration)** - Easiest to set up, high token load, clear validation
2. **Scenario 3 (API Integration)** - Good perturbation design, tests error handling
3. **Scenario 4 (Merge Conflict)** - Novel scenario, agents genuinely struggle with this

These 3 scenarios would give you:
- Variable context sizes (5K → 20K tokens)
- Different failure modes (data issues, API changes, code conflicts)
- Clear validation criteria for success/failure
- Natural SCR spike points at perturbations

---

## Setup Requirements

Each scenario needs:
1. **Seed files** - The initial workspace state
2. **Perturbation data** - New files or modified mock responses
3. **Validator** - Python function that checks success criteria
4. **Ground truth** - Expected final state for RDI calculation
