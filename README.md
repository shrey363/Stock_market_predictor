# Geopolitical Event-Aware Stock Market Prediction and Trading

> **AI/ML Mini Project · FinTech · Team of 2**
> **Status:** Phases 1–6 complete; Phase 7 API/dashboard implemented with its 5 focused tests passing, pending the final project-venv suite/browser gate. Honest results: the rule classifier's balanced accuracy is **0.423**; market+news **0.514** vs. market-only **0.506** is no distinguishable skill; momentum loses **33.13%**; PPO averages **−4.94%** on held-out data with seed returns from **+0.84% to −8.14%**, so it does not reliably beat cash. See [docs/ROADMAP.md](docs/ROADMAP.md) for live verification state and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design.
> **Disclaimer:** This is a research and *simulated* trading prototype. It does not provide financial advice, does not execute real trades, and makes no claim of reliable price prediction.

## Overview

Stock-market movements are shaped by price history, economic conditions, sentiment, and geopolitical developments — armed conflicts, trade restrictions, sanctions, diplomatic decisions, and supply-chain disruptions. This project builds a pipeline that combines historical market data with geopolitical-news analysis:

1. **NLP** extracts structure from news text — event categories and financial sentiment.
2. **Supervised ML** predicts the next session's market direction, first from market data alone, then with geopolitical-news features added, so the two can be compared fairly.
3. **Reinforcement Learning** trains an agent to choose `buy` / `sell` / `hold` inside a historically replayed, cost-aware trading simulator.
4. **A dashboard** exposes news, predictions, model comparisons, and simulated portfolio performance.

The core research question is deliberately modest: *does adding structured geopolitical-news features improve next-session movement prediction over a market-only baseline, on later unseen data?* The answer may well be "no" — negative results get documented too.

## Key principles

- **Small and correct first.** One index or a handful of stocks, a limited event taxonomy, a small historical news dataset. Expand only after each stage is tested.
- **No look-ahead.** Chronological train/validation/test splits, features built only from information available before the prediction cutoff, news published after the cutoff excluded.
- **Honest evaluation.** Naive baselines, precision/recall/F1/balanced accuracy alongside accuracy, documented failed experiments and limitations.
- **Simulation only.** No live or real-money trading, ever.

## Event categories (initial taxonomy)

`armed_conflict` · `trade_restriction` · `sanctions` · `diplomatic_development` · `energy_or_supply_disruption` · `other_or_uncertain`

## Technology stack

| Layer | Choice | Purpose |
|---|---|---|
| Language | Python | Main project language |
| Data handling | Pandas, NumPy | Cleaning and feature engineering |
| Market data | `yfinance` / documented historical dataset | Historical OHLCV |
| Baseline ML | Scikit-learn | Classification and evaluation |
| NLP | Hugging Face Transformers (optionally FinBERT) | Sentiment and text classification |
| RL environment | Gymnasium | Trading environment definition |
| RL algorithms | Stable-Baselines3 | Agent training and evaluation |
| Backend | Flask | API for data and model results |
| Frontend | React + Plotly | Interactive dashboard and charts |
| Storage | CSV/JSON (MongoDB only if needed) | Datasets, metadata, results |

## Repository structure

```text
geopolitical-stock-predictor/
├── README.md
├── PROJECT_DESCRIPTION.md      # Full original specification
├── AGENTS.md                   # Instructions for coding agents (read this first)
├── .gitignore
├── requirements.txt            # Pinned dependencies (verified on Python 3.13)
├── pyproject.toml              # Package metadata + pytest configuration
├── configs/
│   └── config.example.yaml     # Instrument, window, news dataset, provenance, seeds
├── data/
│   ├── manifest.json           # Committed provenance: source, licence, revision, SHA-256
│   ├── raw/                    # Downloads, stored unmodified (git-ignored)
│   ├── interim/                # Filtered news subset (git-ignored, regenerable)
│   └── processed/              # Cleaned, validated tables (git-ignored, regenerable)
├── src/stock_project/
│   ├── common/                 # config.py, paths.py, quality.py, manifest.py, timeutils.py
│   ├── data/                   # market_data.py, news_data.py, fetch_data.py, inspect_data.py
│   ├── preprocessing/          # clean_market_data.py, clean_news_data.py
│   ├── nlp/                    # taxonomy.py, event_classifier.py, sentiment.py,
│   │                           # classify_news.py, annotate.py, labeller.py, evaluate_classifier.py
│   ├── features/               # technical_indicators.py, chronological_splits.py,
│   │                           # news_features.py (Phase 4 pre-cutoff aggregation)
│   ├── models/                 # train_baseline.py (Phase 3), train_news_aware.py (Phase 4)
│   ├── trading/                # simulator.py, gym_env.py, train_rl.py (Phases 5–6)
│   └── backend/                # read-only Flask API + dashboard server (Phase 7)
├── frontend/
│   └── index.html               # React + Plotly dashboard, served by Flask
├── tests/
│   ├── conftest.py             # Offline fixtures (synthetic, hand-checkable)
│   ├── fixtures/               # Tiny CSV/JSONL fixtures used by the tests
│   └── test_*.py               # 19 modules: config, ingestion, cleaning, inspection, nlp,
│                               # features, baselines, news comparison, simulator
├── reports/
│   ├── experiment_results/     # phase1_data_report.json, phase2_*.json,
│   │                           # phase3_baseline_report.json, phase3_test_predictions.csv,
│   │                           # phase4_news_comparison_report.json, phase4_test_predictions.csv,
│   │                           # phase5_simulation_report.json, phase5_portfolio_values.csv
│   └── figures/
└── docs/
    ├── ARCHITECTURE.md         # System design: data flow, layers, invariants, contracts
    ├── ROADMAP.md              # Live progress log — updated with every change
    ├── dataset_notes.md        # Dataset provenance, licences, coverage, gaps
    ├── annotation_guidelines.md # Event taxonomy, tie-breaks, labelling workflow (Phase 2)
    └── model_evaluation.md     # (planned, Phase 3+)
```

*(The implemented prototype uses CSV/JSON artifacts, a Flask backend, and a single-file React + Plotly frontend. No notebook or Node build chain is required.)*

## Development phases

Full detail, checklists, and the running change log live in **[docs/ROADMAP.md](docs/ROADMAP.md)**; the target system design lives in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

| Phase | Goal | Deliverable | Status |
|---|---|---|---|
| 1 | Define and inspect data | Documented, reproducible sample datasets | **Done** — see [docs/dataset_notes.md](docs/dataset_notes.md) |
| 2 | News classification and sentiment | News dataset with traceable category/sentiment outputs | **Done** — evaluated on 200 human labels; balanced accuracy 0.423 / kappa 0.270 (weak on rare categories) |
| 3 | Market features and baseline prediction | Reproducible baseline metrics | **Done** — [phase3_baseline_report.json](reports/experiment_results/phase3_baseline_report.json); no skill over the naive baseline |
| 4 | Add geopolitical features | Fair market-only vs market+news comparison | **Done** — [phase4_news_comparison_report.json](reports/experiment_results/phase4_news_comparison_report.json); news lifts balanced accuracy by 0.008, no distinguishable skill |
| 5 | Trading simulator | Tested historical simulator | **Done** — deterministic long-only engine; buy-and-hold +42.53%, momentum-5 −33.13% after 323 trades and 31,227.96 costs |
| 6 | Reinforcement learning | RL evaluation report with metrics and limitations | **Done** — all 15 checks pass; mean PPO return −4.94%, large seed spread |
| 7 | Dashboard and final integration | Working demo and final report | **In verification** — API/dashboard implemented; focused 5-test contract passes |

## Getting started

Requires **Python 3.11+** (developed and verified on 3.13; every dependency is pinned in `requirements.txt`).

```bash
# 1. Clone and enter the repository
git clone <repo-url>
cd geopolitical-stock-predictor

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies and the package itself
pip install -r requirements.txt
pip install -e .                 # makes `python -m stock_project...` work; or set PYTHONPATH=src
```

### Running Phase 1 (done)

```bash
# Fetch the datasets and cache them under data/raw/ + data/interim/ (needs network, ~84 MB)
python -m stock_project.data.fetch_data --config configs/config.example.yaml

# Clean, validate, and write the Phase 1 report (offline; reads the cache)
python -m stock_project.data.inspect_data --config configs/config.example.yaml
```

`fetch_data` prints the cached file paths and records source/licence/revision/SHA-256 in
`data/manifest.json`. `inspect_data` prints ten pass/fail checks, writes the cleaned tables to
`data/processed/`, and writes [`reports/experiment_results/phase1_data_report.json`](reports/experiment_results/phase1_data_report.json).
Raw and derived datasets are git-ignored because they are fully regenerable with those two
commands; the manifest and the report are committed.

### Running Phase 2 (done)

```bash
# Classify events and score sentiment for the cleaned news table (offline)
python -m stock_project.nlp.classify_news --config configs/config.example.yaml

# Build the review sheets for hand labelling (labels are left empty on purpose)
python -m stock_project.nlp.annotate --config configs/config.example.yaml

# Human step: label the sheet. This serves a local page (127.0.0.1:8391) with one
# headline at a time: press 1-6 for a category, s to skip, u to undo. Labels are
# written to the sheet after every keystroke-confirmed choice, next to an append-only
# audit log. --reviewer records who is labelling; --check prints progress only.
# (Already done: the 200-row sample is fully labelled by a human, so this is only
# needed if you re-generate the sheet.)
python -m stock_project.nlp.labeller --config configs/config.example.yaml --reviewer "your name"
# (rules and allowed values: docs/annotation_guidelines.md)

# Evaluate the baseline against those labels (offline) — writes accuracy, balanced
# accuracy, macro F1, Cohen's kappa and the misclassification list
python -m stock_project.nlp.evaluate_classifier --config configs/config.example.yaml
```

The labeller binds loopback only, never shows the classifier's prediction (labelling stays
blind), rejects anything outside the taxonomy, and skips stay blank — an unreviewed row is
never counted as a correct one. On the 200 labelled rows the baseline scores accuracy 0.905
(naive 0.900), balanced accuracy 0.423, macro F1 0.468 and kappa 0.270: accurate on the majority
`other_or_uncertain` class, weak on every rare category.

### Running Phase 3 (done)

```bash
# Build market features, split chronologically, train the naive and market-only
# baselines, and write the artifacts (offline; reads the Phase 1 cleaned table)
python -m stock_project.models.train_baseline --config configs/config.example.yaml
```

Writes `data/processed/features_SPY_1d.csv` (also recorded in `data/manifest.json`),
[`reports/experiment_results/phase3_baseline_report.json`](reports/experiment_results/phase3_baseline_report.json)
and `phase3_test_predictions.csv`. Every feature in the row for session `t` uses bars only up to
`t-1`; the target is that session's own direction, so the split needs no purge gap. The model's C is
chosen on validation data only and the test period is evaluated once.

Concrete commands for later stages will be added here as those stages land (tracked in the roadmap).

### Running Phase 4 (done)

```bash
# Aggregate pre-cutoff news into per-session features, then train the market-only
# and market+news models on the same chronological split and compare them (offline;
# reads the Phase 1 cleaned market table and the Phase 2 classified news table)
python -m stock_project.models.train_news_aware --config configs/config.example.yaml
```

Writes `data/processed/features_SPY_1d_news.csv` (market + 13 news features, also recorded in
`data/manifest.json`), [`reports/experiment_results/phase4_news_comparison_report.json`](reports/experiment_results/phase4_news_comparison_report.json)
and `phase4_test_predictions.csv` with both models' probabilities for every held-out session.
News features aggregate only articles whose `available_from` falls in
`(news_cutoff - news_lookback_days, news_cutoff]`; sessions with no article get zero counts plus
`no_news = 1`, and sentiment is summed (never imputed as neutral). Both models use the same split,
the same C grid, and the same test rows as Phase 3.

### Running Phase 5 (done)

```bash
# Simulate buy-and-hold and the 5-session momentum benchmark over the cleaned
# market history with the deterministic simulator (offline; reads the Phase 1 table)
python -m stock_project.trading.simulator --config configs/config.example.yaml
```

Writes [`reports/experiment_results/phase5_simulation_report.json`](reports/experiment_results/phase5_simulation_report.json)
and `phase5_portfolio_values.csv` (one equity row per strategy per session). Rules are fixed and
recorded in the artifact: actions are decided pre-open, filled at the session's **open**, and
marked at the **close**; long-only, one position, no pyramiding; a buy spends `position_fraction`
of equity in whole shares capped by cash; costs are charged on both sides at
`transaction_cost_bps`; an unaffordable or redundant action does nothing. The engine is pure
arithmetic, so identical prices and actions always reproduce identical cash and holdings.

### Running Phase 6 (done)

```bash
python -m stock_project.trading.train_rl --config configs/config.example.yaml
```

Writes `phase6_rl_report.json` and `phase6_test_portfolio_values.csv`. Candidate selection uses train/validation only; final PPO agents are evaluated once per seed on the held-out test period.

### Running Phase 7

```bash
python -m stock_project.backend.app --config configs/config.example.yaml
# open http://127.0.0.1:5000
```

The Flask application exposes `/market`, `/news`, `/predictions`, `/models`, `/simulation`, and `/methodology`, and serves the React + Plotly dashboard at `/`. It is loopback-only and read-only: every displayed number comes from a saved CSV/JSON artifact, and no request trains a model or runs a simulation. React and Plotly are pinned CDN assets, so charts require browser internet access; API reads and tests are offline.

## Testing

**Current state:** the last complete project-venv run before Phase 7 was **250 passing**. Phase 7 adds **5 focused tests, all passing**; rerun the full suite in `.venv` before final release. Coverage includes configuration, ingestion, cleaning, NLP, leakage-safe features, chronological supervised comparisons, deterministic trading, the Gymnasium/PPO protocol, and exact agreement between API values and saved artifacts. No test touches the network.

```bash
pytest          # or: pytest -q
```

The specification requires, at minimum: date/timezone parsing checks, duplicate detection, missing-data reporting, leakage guards on features and cutoffs, chronological split verification, simulator cash/position/cost correctness, and agreement between dashboard values and saved experiment results. Tests appear alongside the code they cover.

The suite must run offline and deterministically — no test may download market or news data — and this section is kept in sync with it (how to run, how many tests, what they cover) as each phase lands.

What the suite covers:

| Module | Coverage |
|---|---|
| `tests/test_config.py` | Config loading and strict validation (missing keys, inverted windows, non-UTC timezones, ambiguous dates, out-of-range numbers) |
| `tests/test_market_data.py` | Market ingestion with an injected fake provider: column normalisation, cache reuse, provenance manifest, offline error paths |
| `tests/test_news_data.py` | News download provenance, streaming JSONL filters, deterministic capping, round-trips through disk |
| `tests/test_clean_market_data.py` | Timestamp/timezone normalisation, duplicate detection, chronological ordering, missing-value reporting, no-imputation guarantee |
| `tests/test_clean_news_data.py` | Derived article ids, day-precision `available_from` cutoff guard, duplicate detection, unassigned labels stay unassigned |
| `tests/test_inspect_data.py` | End-to-end run of the inspection CLI on cached fixtures: report contents, processed tables, manifest entries, exit code |
| `tests/test_taxonomy.py` | The six categories match the specification exactly; lexicons are defined, duplicate-free, and polarity-disjoint |
| `tests/test_event_classifier.py` | One expectation per category, priority tie-breaks, negation/conditional flags, veto phrases, scopes, determinism |
| `tests/test_sentiment.py` | Exact hand-computed scores, negation flipping, bounds, mode selection (lexicon works, finbert fails loudly) |
| `tests/test_classify_news.py` | Classified-table contract, traceability columns survive, distribution and report contents, empty-headline handling |
| `tests/test_annotate.py` | Sheet determinism, blank labels, blind by default, focus-sheet strata, and refusal to overwrite human labels |
| `tests/test_evaluate_classifier.py` | Hand-computed precision/recall/F1, balanced accuracy, kappa, naive baseline, label validation, unreviewed handling |
| `tests/test_labeller.py` | Interactive labeller: taxonomy guard, atomic sheet rewrites, undo, audit log, blind browser payload, served-page JS escape regression, HTTP round-trip, loopback-only binding |
| `tests/test_technical_indicators.py` | Hand-computed returns, SMA distance, Wilder RSI (incl. flat/rising/falling edge cases), volatility, volume ratio, range/gap; the cutoff guard (future prices cannot change a row, history must), and the feature-table contract |
| `tests/test_chronological_splits.py` | Contiguous ordered ranges, deterministic floors, invalid fractions rejected, periods time-ordered and disjoint |
| `tests/test_train_baseline.py` | Hand-computed precision/recall/F1/balanced accuracy and AUC, the artifact contract (report, predictions, manifest), determinism, and missing-input error path |
| `tests/test_news_features.py` | News window `(cutoff - lookback, cutoff]`, day-precision availability, per-category/sentiment counts, explicit no-news rows (no imputed sentiment), recency capping, post-cutoff leakage guard, taxonomy/schema hard errors |
| `tests/test_train_news_aware.py` | Market-only metrics reproduce Phase 3 exactly on the same split, both models share the same held-out rows, combined feature-store and comparison artifact contracts, determinism, CLI exit status |
| `tests/test_trading_simulator.py` | Hand-computed buy/sell/hold cash, shares, notional, costs and portfolio values; no overspending or shorting; chronological-order refusal; momentum reads no same-session price; deterministic replay; benchmark and equity artifact contracts |
| `tests/test_gym_env.py`, `tests/test_train_rl.py` | Gymnasium contract, information-set leakage, simulator agreement, held-out PPO protocol, risk metrics and Phase 6 artifacts |
| `tests/test_backend.py` | Six read-only API contracts, exact saved-artifact values, pagination/filter validation, missing-artifact errors, and required dashboard/responsible-use content |

## Limitations and responsible use

- Historical patterns may not persist; backtests and RL agents can overfit historical paths and reward choices.
- News can be incomplete, delayed, biased, or wrong; sentiment does not determine return direction.
- Geopolitical events affect sectors differently and may already be priced in.
- Transaction costs, liquidity, slippage, and execution timing materially change simulated results.
- Small datasets rarely represent rare, unprecedented events.

Treat every number this project produces as a model output for study — never as financial advice.

## References

- [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md) — full specification, schema, milestones, and risk register.
- [docs/ROADMAP.md](docs/ROADMAP.md) — step-by-step progress and change log.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — target architecture: data flow, layers, design invariants, API surface.
- [AGENTS.md](AGENTS.md) — operating rules for coding agents: doc-update duties, testing and data conventions, guardrails.
