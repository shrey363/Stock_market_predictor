# AGENTS.md — Instructions for Coding Agents

This file is the entry point for any AI coding agent working on this repo.
Read it before writing code, and follow it strictly.

## 1. What this project is

**Geopolitical Event-Aware Stock Market Prediction and Trading** — a research prototype (team of 2, FinTech AI/ML mini-project) that:

1. Extracts structure from geopolitical news with NLP (event category + financial sentiment).
2. Predicts next-session market direction with supervised ML — market-only baseline vs. market+news — compared fairly.
3. Trains an RL agent (`buy` / `sell` / `hold`) inside a historically replayed, cost-aware trading simulator.
4. Exposes everything in a dashboard.

Core research question: *does adding structured geopolitical-news features improve next-session prediction over a market-only baseline on later unseen data?* A "no" is a valid, documentable result.

> **Simulation only. No live or real-money trading, ever. No financial advice. Never claim reliable prediction or profit.**

**Current state:** documentation only — `README.md`, `PROJECT_DESCRIPTION.md`, `AGENTS.md`, `docs/ROADMAP.md`, `docs/ARCHITECTURE.md`. There is **no Python code, no `requirements.txt`, no `tests/`, and no test suite yet**; all six Phase 1 items are unchecked. Do not describe the project as tested or working. Refresh this line whenever the state changes.

## 2. Source-of-truth documents (read before acting)

| Document | Role |
|---|---|
| `PROJECT_DESCRIPTION.md` | Full specification — schema, milestones (§14), testing (§15), risks (§16), Definition of Done (§19). |
| `README.md` | Overview, principles, taxonomy, stack, target repo structure, phase table. |
| `docs/ROADMAP.md` | **Live progress tracker.** Phase checklists + change log. Tells you what is done and what is next. |
| `docs/ARCHITECTURE.md` | Target design — data flow, layer responsibilities, invariants, data contracts, planned Flask API, evolution plan. |
| `AGENTS.md` | This file — operating rules for agents. |

**Precedence when documents disagree:**

- `PROJECT_DESCRIPTION.md` wins on **what to build** (schema, scope, milestones, evaluation). Never edit it unilaterally — flag the discrepancy in `docs/ROADMAP.md` and ask.
- `AGENTS.md` wins on **how to work** (process, testing, doc duties).
- `docs/ROADMAP.md` is the truth about **what is actually done and verified**. If the code and the roadmap disagree, the code is wrong or unverified until the roadmap is corrected.

Always check `docs/ROADMAP.md` → Overall status + phase checklists before starting work so you don't redo done work or skip ahead.

## 3. MANDATORY: update documents at every single stage of development

**Nothing counts as done until the docs say so.** This is the project's strictest rule.

On **every** substantive change — code, data, config, experiment, fix, decision, or doc — you MUST:

1. **`docs/ROADMAP.md`**
   - Tick the checklist box only when the work is done **and verified** (tests written, run, and passing; outputs inspected).
   - Add a change-log entry at the top (newest first) with: date, what changed, files touched, tests added/run, and verification performed.
   - Update the `Overall status` / current-state line to state the test count and project state, e.g.: `Overall status: Phase 2 — X/Y tests passing (pytest ...), <one-line state>`. While no suite exists, say so explicitly rather than showing a misleading `0/0 passing`.
2. **`README.md`** — update if setup, repo structure, run commands, testing, or phase summary changed. Keep its Testing section in sync with the current suite (how to run, how many tests, what they cover) and with the current repo structure tree.
3. **`docs/ARCHITECTURE.md`** — update if data flow, module responsibilities, invariants, contracts, or API surface changed.
4. **`AGENTS.md`** — update if the process, conventions, or repo state above changed. A change to this file is itself a change; log it in the change log like any other.
5. **`PROJECT_DESCRIPTION.md`** — do NOT edit spec unilaterally; if work contradicts it, flag the discrepancy instead.

If your change is small but meaningful (bug fix, metric update, config tweak), at minimum add a `docs/ROADMAP.md` change-log entry. Never batch doc updates "for later".

Keep documentation honest and current: no aspirational claims ("implemented", "tested", "passing") for anything not actually run. Planned items stay labelled planned.

## 4. Non-negotiable engineering invariants

From `docs/ARCHITECTURE.md` §3 — violating these is a bug regardless of metrics:

1. **Chronological everything.** Time-ordered train/validation/test splits. No shuffling. Fit transforms on train only.
2. **Prediction cutoff.** Every feature row uses only information strictly before prediction time. Post-cutoff news is excluded, not down-weighted.
3. **No silent fills.** Missing news → zero counts + explicit `no_news` indicator. Never impute neutral sentiment.
4. **Traceability.** Every classified article keeps `article_id`, original text, timestamp, source, category, sentiment.
5. **Simulator determinism.** Same prices + costs + actions → identical results. Unit-test with hand-checkable examples before RL.
6. **RL information set.** Observations contain only data available at simulated time. No future leakage.
7. **File-based artifacts.** Models/metrics/runs go to `reports/experiment_results/`. Backend/dashboard only read them.

Plus project principles (`README.md`): **small and correct first** (one index / few stocks, 6 event categories, small dataset — expand only after tests pass); **honest evaluation** (naive baselines, precision/recall/F1/balanced accuracy, document failures); **no scope creep** (finish the current phase deliverable first).

Event taxonomy: `armed_conflict` · `trade_restriction` · `sanctions` · `diplomatic_development` · `energy_or_supply_disruption` · `other_or_uncertain` — never force an uncertain headline into a confident category.

## 5. How to work

1. **Pick work from the roadmap.** Do the earliest unchecked Phase 1–7 item. Do not jump to RL (Phase 6) or dashboard (Phase 7) before earlier phases are verified. Guiding order is in `PROJECT_DESCRIPTION.md` §20.
2. **Keep the stack small.** Python, Pandas/NumPy, `yfinance` or documented dataset, scikit-learn, Hugging Face Transformers (FinBERT optional), Gymnasium, Stable-Baselines3, Flask, React + Plotly, CSV/JSON (MongoDB only if justified). Add deps only when needed; pin them in `requirements.txt` once Phase 1 lands.
3. **Grow the repo lazily.** Target layout is in `README.md`. Don't scaffold empty complexity — create `configs/`, `data/`, `notebooks/`, `src/stock_project/`, `backend/`, `frontend/`, `tests/`, `reports/` files only when the active phase needs them. Never commit `data/raw/`.
4. **Create tests at every step — no exceptions.** Every phase/stage must add or extend tests under `tests/` covering the new code (naming: `test_<area>.py`). Minimum per spec §15: date/timezone parsing, duplicates, missing-data reporting, leakage guards (features + cutoffs), chronological-split verification, simulator cash/position/cost correctness, dashboard-vs-artifact agreement. Run the full suite with `pytest` (or `pytest -q`) and record the exact result (e.g. `12 passed, 0 failed`) in the `docs/ROADMAP.md` change-log entry and `Overall status` line. A stage with no new/updated tests is incomplete.
5. **Report honestly.** Negative results, class imbalance, misclassifications, and limitations go into the docs and final report — not hidden.
6. **Keep changes reviewable.** One focused change per task; don't refactor or restructure files you were not asked to touch. Preserve the existing file conventions (the working tree is CRLF; keep edits CRLF-consistent so diffs stay minimal).

## 6. Environment and commands

Python 3.11+ is assumed (3.13 is installed on the current dev machine). Nothing is installed or pinned yet.

```bash
# once Phase 1 lands: environment + dependencies
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# tests (once a suite exists)
pytest -q
```

- Prefer `python -m pytest` style invocation only if the plain `pytest` on PATH is the wrong interpreter; otherwise keep commands exactly as documented in `README.md`.
- Network access is a **soft** dependency: `yfinance` needs it for ingestion, but tests and normal verification must not. Cache raw downloads under `data/raw/` (git-ignored) and never let a test fail because the network is down.
- If you add, remove, or change a dependency, update `requirements.txt` (pinned) and the README setup section in the same change.

## 7. Testing conventions

- Tests live in `tests/`, one file per area (`test_data_validation.py`, `test_features.py`, `test_trading_simulator.py`, …), mirroring spec §15 rather than file count.
- Tests must be **offline and deterministic**: set seeds/fixtures (or monkeypatch data loaders) so results never depend on wall-clock time, network, or run order. No test may download market or news data.
- Timestamps are timezone-aware and normalised consistently (store/normalise to UTC); a test must cover the date/timezone parsing rule being used.
- Prefer hand-checkable numeric fixtures over snapshots, especially for the simulator: assert exact cash, positions, portfolio value, and costs on tiny hard-coded price series.
- Every leakage guard must be a real assertion (e.g. a mutated future row must not change a feature value), not a comment.
- Run the **full** suite before claiming done and quote the real output line. If tests fail, fix the cause; never weaken an assertion, skip a test, or add a lint/type suppression to make a check pass. If a check genuinely cannot run, report it as not run — never as passing.
- Documentation-only changes add no tests by themselves, but must still record in the change log that no suite exists / what was verified instead.
- Experiment outputs (metrics, figures, run configs) are saved under `reports/experiment_results/` (and `reports/figures/`), not just printed.

## 8. Data and licensing rules

- Never invent data: no fabricated timestamps, labels, prices, sector links, or source attributions. If a field is unknown, leave it missing and report it.
- Every dataset gets recorded in `docs/dataset_notes.md`: source, licence/terms, retrieval date, columns, date coverage, known gaps, and any preprocessing applied.
- Check licence and access terms **before** adding a dataset or API, and note the conclusion. If terms are unclear or restrictive, stop and ask.
- Raw data is never committed (`data/raw/` is git-ignored); derived artifacts must be small and deterministically regenerable from documented sources.
- Duplicates, missing values, and timezone ambiguity are reported (counts + affected rows), never silently dropped or imputed.

## 9. Definition of Done (per change)

A change is complete only when ALL of the following hold (for a docs-only change, item 1 is satisfied by logging why no tests apply):

- [ ] New/updated tests in `tests/` cover the change; full suite run via `pytest` with counts recorded (e.g. `X passed, 0 failed`) — no test-less stages.
- [ ] No invariant in §4 is violated (cutoff, chronology, leakage, determinism checked).
- [ ] The behavior was verified through the artifact the user/next phase consumes (a run command, a saved file, the dashboard value) — not just a clean exit code.
- [ ] `docs/ROADMAP.md` checklist + change log updated (date, files, tests added, `X/Y tests passing`, verification).
- [ ] `Overall status` line in `docs/ROADMAP.md` restates current test state and project state (e.g. `Phase N — X/Y tests passing, <state>`).
- [ ] `README.md` / `docs/ARCHITECTURE.md` / `AGENTS.md` updated if affected.
- [ ] Experiment numbers (if any) are saved as reproducible artifacts under `reports/`, not just printed to console.
- [ ] Cross-links between docs still resolve after any rename or move.

## 10. Guardrails: never do this

- Never add, simulate, or describe live/real-money trading, broker connections, or order routing — including "just a demo" code paths. Simulation only.
- Never claim profitability, reliable prediction, or that a result is validated when it is not backtested/held-out.
- Never edit `PROJECT_DESCRIPTION.md` (or quietly redefine the research question, target, or taxonomy) to fit an implementation.
- Never commit `data/raw/`, secrets, tokens, large binaries, or `.venv/`.
- Never commit, push, or open a PR unless the user asks. Never run destructive or history-rewriting git commands, and never discard work you did not create.
- Never expand scope (extra tickers, extra event categories, extra models, MongoDB, Streamlit, deep learning) before the current phase's deliverable is tested — propose it instead.
- Never hide a failure: if a stage's numbers are bad or a check could not run, say so in the docs and in the reply.

## 11. Quick agent checklist (copy into each task)

- [ ] Read `docs/ROADMAP.md` status + relevant spec section.
- [ ] Implement the smallest correct step for the current phase.
- [ ] Add/update tests in `tests/`; run full `pytest` suite and note `X passed / Y failed`.
- [ ] Save artifacts to `reports/experiment_results/` where applicable.
- [ ] Update `docs/ROADMAP.md` (tick + change log with test counts + `Overall status: ... X/Y tests passing, <state>`) **in the same task**; update `README.md` / `ARCHITECTURE.md` / `AGENTS.md` if affected.
- [ ] Report back: what changed and why, files touched, verification command + its real output, docs updated, and anything not done or not verifiable.
