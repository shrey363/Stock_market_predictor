# Geopolitical Event-Aware Stock Market Prediction and Trading

> **AI/ML Mini Project · FinTech · Team of 2**
> **Status:** Phase 1 complete; Phase 2 implemented (event classifier + sentiment baseline) with evaluation **pending the hand-labelled review sample** (label it with the bundled labeller, `python -m stock_project.nlp.labeller`) — 178 tests passing. See [docs/ROADMAP.md](docs/ROADMAP.md) for live progress, [docs/annotation_guidelines.md](docs/annotation_guidelines.md) for the labelling rules, [docs/dataset_notes.md](docs/dataset_notes.md) for the datasets, and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design.
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
│   ├── features/               # (planned, Phase 3) technical_indicators.py, news_features.py
│   ├── models/                 # (planned, Phase 3-4) train_baseline.py, train_news_aware.py
│   └── trading/                # (planned, Phase 5-6) simulator.py, gym_env.py, train_rl.py
├── tests/
│   ├── conftest.py             # Offline fixtures (synthetic, hand-checkable)
│   ├── fixtures/               # Tiny CSV/JSONL fixtures used by the tests
│   └── test_*.py               # 13 modules: config, ingestion, cleaning, inspection, nlp
├── reports/
│   ├── experiment_results/     # phase1_data_report.json, phase2_*.json, misclassifications
│   └── figures/
└── docs/
    ├── ARCHITECTURE.md         # System design: data flow, layers, invariants, contracts
    ├── ROADMAP.md              # Live progress log — updated with every change
    ├── dataset_notes.md        # Dataset provenance, licences, coverage, gaps
    ├── annotation_guidelines.md # Event taxonomy, tie-breaks, labelling workflow (Phase 2)
    └── model_evaluation.md     # (planned, Phase 3+)
```

*(Phase 1 created the parts shown without a `(planned…)` marker. `notebooks/`, `backend/`,
and `frontend/` arrive in later phases, as does the rest of `src/stock_project/`.)*

## Development phases

Full detail, checklists, and the running change log live in **[docs/ROADMAP.md](docs/ROADMAP.md)**; the target system design lives in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

| Phase | Goal | Deliverable | Status |
|---|---|---|---|
| 1 | Define and inspect data | Documented, reproducible sample datasets | **Done** — see [docs/dataset_notes.md](docs/dataset_notes.md) |
| 2 | News classification and sentiment | News dataset with traceable category/sentiment outputs | **Implemented** — evaluation pending hand labels |
| 3 | Market features and baseline prediction | Reproducible baseline metrics | Planned |
| 4 | Add geopolitical features | Fair market-only vs market+news comparison | Planned |
| 5 | Trading simulator | Tested historical simulator | Planned |
| 6 | Reinforcement learning | RL evaluation report with metrics and limitations | Planned |
| 7 | Dashboard and final integration | Working demo and final report | Planned |

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

### Running Phase 2 (implemented; evaluation needs human labels)

```bash
# Classify events and score sentiment for the cleaned news table (offline)
python -m stock_project.nlp.classify_news --config configs/config.example.yaml

# Build the review sheets for hand labelling (labels are left empty on purpose)
python -m stock_project.nlp.annotate --config configs/config.example.yaml

# Human step: label the sheet. This serves a local page (127.0.0.1:8391) with one
# headline at a time: press 1-6 for a category, s to skip, u to undo. Labels are
# written to the sheet after every keystroke-confirmed choice, next to an append-only
# audit log. --reviewer records who is labelling; --check prints progress only.
python -m stock_project.nlp.labeller --config configs/config.example.yaml --reviewer "your name"
# (rules and allowed values: docs/annotation_guidelines.md)

# Evaluate the baseline against those labels (offline)
python -m stock_project.nlp.evaluate_classifier --config configs/config.example.yaml
```

The labeller binds loopback only, never shows the classifier's prediction (labelling stays
blind), rejects anything outside the taxonomy, and skips stay blank — an unreviewed row is
never counted as a correct one.

Concrete commands for later stages will be added here as those stages land (tracked in the roadmap).

## Testing

**Current state:** 178 tests, all passing (`178 passed` via `pytest`), covering configuration
validation, market and news ingestion, cleaning/validation, the Phase 1 inspection artifact,
and the Phase 2 machinery (taxonomy integrity, rule classifier, sentiment baseline, review
sheets, the interactive labeller, metrics). No test touches the network.

```bash
pytest          # or: pytest -q
```

The specification requires, at minimum: date/timezone parsing checks, duplicate detection, missing-data reporting, leakage guards on features and cutoffs, chronological split verification, simulator cash/position/cost correctness, and agreement between dashboard values and saved experiment results. Tests appear alongside the code they cover.

The suite must run offline and deterministically — no test may download market or news data — and this section is kept in sync with it (how to run, how many tests, what they cover) as each phase lands.

What the current 178 tests cover:

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
