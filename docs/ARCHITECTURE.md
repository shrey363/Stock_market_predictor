# Architecture

Basic system architecture for the **Geopolitical Event-Aware Stock Market Prediction and Trading** project. This describes the *target* design; components are implemented phase by phase per [ROADMAP.md](ROADMAP.md). **Phases 1–5 are implemented** — ingestion, cleaning, inspection, event classification, sentiment scoring, the review/evaluation tooling, market feature engineering, pre-cutoff news aggregation, chronological splitting, the naive plus market-only baselines, the market-plus-news comparison, and the deterministic trading simulator with its benchmarks live in `src/stock_project/` (§10 lists exactly what exists). Phase 2's classifier has been **evaluated against 200 human labels** (weak on rare categories: balanced accuracy 0.423, kappa 0.270). The Gymnasium wrapper and RL training, plus the `backend/` and `frontend/` layers, are still target design.

- Specification: [PROJECT_DESCRIPTION.md](../PROJECT_DESCRIPTION.md)
- Overview and setup: [README.md](../README.md)
- Agent and process rules: [AGENTS.md](../AGENTS.md)

---

## 1. High-level data flow

```text
                        ┌──────────────────────────────────────────────────────────┐
                        │                        DATA LAYER                        │
                        │  data/raw  →  data/interim  →  data/processed            │
                        │  (CSV/JSON: market OHLCV, news articles, features)       │
                        └───────────────┬───────────────────────┬──────────────────┘
                                        │                       │
                 ┌──────────────────────▼───────┐    ┌──────────▼──────────────────┐
                 │      MARKET PIPELINE         │    │        NEWS PIPELINE        │
                 │  market_data.py (yfinance)   │    │  news_data.py (dataset/API) │
                 │  clean_market_data.py        │    │  clean_news_data.py         │
                 │  technical_indicators.py     │    │  event_classifier.py        │
                 │  returns, MA, RSI, volatility│    │  sentiment.py (FinBERT/BL)  │
                 └──────────────┬───────────────┘    └──────────┬──────────────────┘
                                │                               │
                                │               ┌───────────────▼──────────────┐
                                │               │   news_features.py           │
                                │               │   daily aggregation,         │
                                │               │   PRE-CUTOFF articles only   │
                                │               └───────────────┬──────────────┘
                                │                               │
                 ┌──────────────▼───────────────────────────────▼──────────────┐
                 │                    FEATURE STORE (data/processed)           │
                 │  one row per session: market features + news features       │
                 │  + binary target (next-session return > 0)                  │
                 └──────────────┬───────────────────────────────┬──────────────┘
                                │                               │
                 ┌──────────────▼──────────────┐    ┌───────────▼─────────────────┐
                 │   SUPERVISED MODELS         │    │   TRADING SIMULATOR         │
                 │  train_baseline.py          │    │  simulator.py deterministic │
                 │   (naive + market-only)     │    │  gym_env.py (Gymnasium)     │
                 │  train_news_aware.py        │    │  train_rl.py (SB3 agent)    │
                 │  evaluate.py (chronological)│    │  benchmarks: buy&hold, rule │
                 └──────────────┬──────────────┘    └───────────┬─────────────────┘
                                │                               │
                                │        metrics.json /         │
                                │        results CSV/JSON       │
                 ┌──────────────▼───────────────────────────────▼──────────────┐
                 │                        API LAYER (Flask)                    │
                 │   endpoints: /market, /news, /predictions, /models,         │
                 │               /simulation, /methodology                     │
                 └──────────────────────────────┬──────────────────────────────┘
                                                │ JSON
                 ┌──────────────────────────────▼──────────────────────────────┐
                 │                  FRONTEND (React + Plotly)                  │
                 │  Market overview · News explorer · Prediction view ·        │
                 │  Model comparison · Trading simulation · Limitations        │
                 └─────────────────────────────────────────────────────────────┘
```

## 2. Layers and responsibilities

| Layer | Modules (target) | Responsibility |
|---|---|---|
| **Ingestion** | `src/stock_project/data/market_data.py`, `news_data.py` | Fetch/load raw OHLCV and news; record source and retrieval date; never modify raw files |
| **Preprocessing** | `preprocessing/clean_market_data.py`, `clean_news_data.py` | Parse dates and timezones, deduplicate, report missing values, clean text |
| **NLP** | `nlp/taxonomy.py`, `event_classifier.py`, `sentiment.py`, `classify_news.py`, `annotate.py`, `labeller.py`, `evaluate_classifier.py` | Assign event category (rule baseline, priority order, veto phrases) and sentiment label/score; keep every output traceable to `article_id` **and** to the matched terms; produce the human review sheets, collect human labels without inventing any, and evaluate |
| **Features** | `features/technical_indicators.py`, `features/news_features.py` | Session-level market indicators; news aggregation using only pre-cutoff articles; explicit no-news indicator |
| **Models** | `models/train_baseline.py`, `train_news_aware.py`, `evaluate.py` | Naive baseline, market-only and market+news models with matched settings; chronological evaluation |
| **Trading** | `trading/simulator.py`, `gym_env.py`, `train_rl.py` | Deterministic portfolio simulator, Gymnasium wrapper, SB3 training, benchmark strategies |
| **Backend** | `src/stock_project/backend/app.py` (Flask) | Implemented read-only JSON API over saved datasets and experiment results; also serves the dashboard |
| **Frontend** | `frontend/index.html` (React + Plotly) | Implemented single-page dashboard with all six specification sections and accessible table views |
| **Common** | `common/` | Config loading, paths, logging, seeds |

## 3. Key design rules

These are architectural invariants, not suggestions — a component that breaks them is wrong regardless of its metrics.

1. **Chronological everything.** Train/validation/test splits are time-ordered; no random shuffling of series. Fitted transformations are fit on training data only.
2. **Prediction cutoff.** Every feature row is built from information available strictly before the session's prediction time. News published after the cutoff is excluded, not down-weighted. Because the news source carries only a calendar date, the cleaned table exposes `available_from` (= publication date + 1 day) as the earliest instant an article may be used; features must read that column, never `published_at`.
3. **No silent fills.** Missing news → zero counts *plus* a missing-news indicator. Never impute neutral sentiment for absence of data.
4. **Traceability.** Every classified headline keeps its `article_id`, original text, timestamp, source, assigned category, and sentiment score so dashboard values can be traced to inputs.
5. **Simulator determinism.** Given the same prices, costs, and actions, the simulator produces identical results; it is unit-tested with hand-checkable examples before RL touches it.
6. **RL information set.** The Gymnasium observation contains only data available at the simulated time (no future prices, news, or the held-out split).
7. **Experiment artifacts are files.** Models, metrics, and simulation runs are written to `reports/experiment_results/` as reproducible artifacts; the backend and dashboard only *read* them, so displayed numbers always match saved results.

## 4. Data contracts

Full field lists are in §6 of the specification. Summary of the two core tables:

- **Market table (cleaned, implemented):** `symbol, timestamp, open, high, low, close, adj_close, volume, source` — one row per exchange session, timestamp UTC, unique on `(symbol, timestamp)`, never gap-filled or imputed.
- **News table (cleaned, implemented):** `article_id, published_at, available_from, headline, article_text, source, source_url, source_category, authors, event_category, sentiment_label, sentiment_score, affected_sector` — where `article_id` is derived (SHA-1 of `source_url|headline|date`), `source` is the publisher host derived from the URL, `source_category` is the *publisher's* label kept separate from our taxonomy, `available_from = published_at + 1 day`, and the four label columns are left **empty** by cleaning — Phase 2 fills `event_category`, `sentiment_label`, and `sentiment_score` in the separate classified table below, and `affected_sector` stays unassigned.
- **Filtered news subset (interim, implemented):** `link, headline, category, short_description, authors, date` — the source columns for one publisher category inside the configured window, before validation.
- **Classified news table (processed, implemented):** the Phase 1 contract plus `event_category, event_priority, matched_keywords, competing_categories, excluded_evidence, evidence_negated, conditional, classified_scope, matched_term_count, sentiment_label, sentiment_score, sentiment_terms`. Machine-generated baseline outputs — never presented as human labels. `event_category` is written by the classifier; the *human* label column lives only in `data/annotations/`.
- **Review sheets (annotations, implemented):** `article_id, published_at, headline, source, source_category, sampling_reason, event_category, reviewer, reviewed_at, notes` — the label column is written empty and filled only by a human. `data/annotations/` is committed because hand labelling is expensive work that must not be lost or overwritten.
- **Feature table (processed, implemented):** `symbol, session_date, decision_time_utc, news_cutoff_utc`, market features, news features, then `next_session_return, target`. One row per predicted session. Market features (Phase 3): `return_{1,5,20}d, sma_ratio_{10,50}, rsi_14, volatility_20, volume_ratio_20, intraday_range, overnight_gap` — each evaluated at the close of its bar and shifted forward one row, so the row for session `t` uses bars up to and including `t-1` only. News features (Phase 4): `news_count_total`, `news_count_<category>` (one per taxonomy category), `news_positive_count`, `news_negative_count`, `news_neutral_count`, `news_sentiment_sum`, `news_days_since_last`, `no_news`. Only articles with `available_from` in `(news_cutoff_utc - news_lookback_days, news_cutoff_utc]` are aggregated; with no article the counts and sentiment sum are zero, `news_days_since_last` is the capped lookback, and `no_news = 1` — sentiment is never imputed as neutral (it is a sum, not a mean, for exactly this reason). `target = 1` when that session closes above the previous close. Warm-up rows are dropped and counted, never filled.

Storage stays CSV/JSON on disk until a real need arises; MongoDB is explicitly a later option, not a default dependency.

## 5. API surface (Implemented, Phase 7)

| Endpoint | Returns |
|---|---|
| `GET /market` | OHLCV + indicators for the configured instrument(s) |
| `GET /news` | Classified headlines with category, sentiment, timestamps |
| `GET /predictions` | Per-session predicted probability/direction + top features |
| `GET /models` | Side-by-side metrics: naive vs market-only vs market+news |
| `GET /simulation` | Portfolio value series, trades, drawdown, benchmark comparison |
| `GET /methodology` | Data sources, cutoff rule, evaluation period, caveats |

The API is read-only and serves only saved artifacts — training runs offline via CLI/notebooks, never inside a request.

## 6. Configuration and reproducibility

- `configs/config.example.yaml` — instrument symbols, date ranges, cutoff time, cost assumptions, model hyperparameters, random seeds.
- Seeds are set centrally (`common/`) and recorded in every experiment artifact.
- `requirements.txt` pins versions once Phase 1 lands; raw data is git-ignored, processed feature sets are small enough to regenerate deterministically from documented sources.

## 7. Evolution plan

The design is intentionally boring and grows per roadmap phase:

- **Phases 1–2:** ingestion + preprocessing + NLP layers only (notebooks drive exploration).
- **Phase 3–4:** feature store and supervised models; comparison artifacts.
- **Phase 5–6:** simulator, Gymnasium env, RL training — all consuming the same feature store.
- **Phase 7:** Flask API and React dashboard bolt *onto* existing artifacts; nothing is rewired.

If a stage's tests are green, later stages can be built without reworking earlier ones. That independence is the main reason for the strict file-based contracts between layers.

## 8. Invariant → test map

Every rule in §3 has (or will have) an explicit, offline test — a comment is not enforcement. This table is the contract between the design and the suite required by §15 of the specification.

| Invariant (§3) | Test that enforces it | Lands in |
|---|---|---|
| Chronological splits | Train/validation/test date ranges are strictly ordered and disjoint; no shuffling anywhere in the path (`test_chronological_splits`, `test_train_baseline`) | Phase 3 — done |
| Prediction cutoff (market) | Mutating prices at or after session `t` leaves row `t`'s features unchanged, while mutating earlier prices does change them (`test_mutating_the_predicted_session_leaves_its_features_untouched`, `test_features_do_depend_on_earlier_prices`) | Phase 3 — done |
| Prediction cutoff (news) | Adding/changing a news row published after the cutoff leaves that session's feature row unchanged (`test_post_cutoff_articles_cannot_change_a_session_row`) | Phase 4 — done |
| No silent fills | Phase 1: per-column missing counts are reported, no value is imputed or forward-filled, and label columns stay empty. Phase 4: a session with no pre-cutoff article gets all-zero counts **and** `no_news = 1`, with no sentiment value imputed (`test_session_without_news_is_explicit_not_imputed`) | Phase 1, completed in Phase 4 — done |
| Day-level news timestamps | Every cleaned article has `available_from = published_at + 1 day`, so a same-day cutoff cannot leak post-cutoff news; the aggregation window is tested at both boundaries (`test_window_is_strictly_after_start_and_inclusive_at_cutoff`, `test_day_precision_news_published_the_day_before_is_usable`) | Phase 1 (guard), consumed and tested in Phase 4 |
| Traceability | Every classified article still carries `article_id`, text, timestamp, and source, plus the matched keywords, competing categories, and vetoed evidence behind its label | Phase 2 (asserted in `test_clean_news_data`, `test_classify_news`) |
| No invented labels | The classifier writes machine labels only into the processed table; the human review sheet's label column is written empty, unreviewed rows are counted separately from correct ones, and a label outside the taxonomy is a hard error. The labeller only ever writes what a reviewer pressed, keeps skips blank, logs every change to an append-only JSONL file, and serves the reviewer context columns only — never the prediction | Phase 2 (`test_annotate`, `test_evaluate_classifier`, `test_labeller`) |
| Simulator determinism | Replaying identical prices + costs + actions yields identical cash, positions, and portfolio value (`test_replay_is_deterministic`, plus hand-checked fixtures) | Phase 5 — done |
| RL information set | Mutating rows after step `t` does not change the observation at `t` | Phase 6 |
| Artifact contract | Values returned by the API/dashboard equal the values in `reports/experiment_results/` | Phase 7 |

Tests run offline and deterministically: no network, no wall-clock dependence, fixed seeds where randomness is involved. Raw data is cached under `data/raw/` (git-ignored) so ingestion can be tested from a local fixture.

## 9. Module boundaries

Dependencies point one way only, left to right:

```text
common  →  data  →  preprocessing  →  nlp / features  →  models / trading  →  backend  →  frontend
```

- `common/` (config, paths, logging, seeds) must not import any other project layer.
- `nlp/` and `features/` may read `data/` and `preprocessing/` outputs, never `models/` or `trading/`.
- `backend/` and `frontend/` read saved artifacts only; they never import training or simulation code and never trigger training inside a request.
- Nothing outside `trading/` maintains portfolio state, and nothing outside `reports/` persists experiment numbers.

Cross-layer imports that skip or reverse this order are a design bug even when the code runs — restructure or route through `common/` instead.

## 10. Implementation status (Phases 1–5)

Implemented and tested:

| Module | Responsibility |
|---|---|
| `common/config.py` | Strict YAML config loading; rejects missing keys, inverted windows, non-UTC timezones, ambiguous dates, out-of-range numbers |
| `common/paths.py` | Every data/report path derived from the repository root, with a test-friendly root override |
| `common/quality.py` | `QualityReport` — missing counts/ratios, duplicates, coerced values, timezone assumptions, warnings |
| `common/manifest.py` | Provenance manifest (`data/manifest.json`): source, licence, revision, retrieval time, rows, coverage, SHA-256 |
| `common/timeutils.py` | UTC clock helpers |
| `data/market_data.py` | Market fetch with an injectable downloader, local cache, and manifest entries |
| `data/news_data.py` | Pinned-revision news download plus a streaming, deterministic JSONL subset filter |
| `data/fetch_data.py` | CLI: fetch both datasets (the only step that needs network) |
| `data/inspect_data.py` | CLI: clean, validate, write `data/processed/` and the Phase 1 JSON report |
| `preprocessing/clean_market_data.py` | Strict ISO-8601 parsing, UTC normalisation, dedupe, ordering, sanity warnings, no imputation |
| `preprocessing/clean_news_data.py` | Derived ids, day-precision `available_from` guard, dedupe, unassigned labels stay unassigned |
| `nlp/taxonomy.py` | The six categories in priority order, keyword lexicons, veto phrases, negation/conditional markers, sentiment terms (versioned data, not logic) |
| `nlp/event_classifier.py` | Transparent rule baseline: token-sequence matching, priority tie-breaks, negation/speculation flags, veto suppression, per-article evidence columns |
| `nlp/sentiment.py` | Lexicon sentiment on `[-1, 1]` with negation flipping; `finbert` is a documented, not-yet-wired mode that fails loudly |
| `nlp/classify_news.py` | CLI: writes `data/processed/news_classified_*.csv` and `phase2_classification_report.json` |
| `nlp/annotate.py` | CLI: deterministic random + stratified review sheets with empty labels; refuses to overwrite existing labels |
| `nlp/labeller.py` | CLI + loopback-only HTTP UI: one headline at a time, keyboard shortcuts, atomic sheet writes, undo, and an append-only `<sheet>_labelling_log.jsonl` audit trail; blind by construction |
| `nlp/evaluate_classifier.py` | CLI: per-class precision/recall/F1, balanced accuracy, kappa, naive baseline, error listing — `phase2_classifier_evaluation.json` |
| `features/technical_indicators.py` | Returns, SMA distance, Wilder RSI, rolling volatility, volume ratio, intraday range, overnight gap — all shifted forward one row so the cutoff is structural; builds the target and writes the feature table |
| `features/news_features.py` | Pre-cutoff news aggregation per session: category counts, positive/negative/neutral counts, sentiment sum, recency (capped), explicit `no_news`; unknown categories/labels and unparsable timestamps are hard errors |
| `features/chronological_splits.py` | Ordered train/validation/test ranges (`int()` floor, deterministic) plus per-period rows, dates and class balance; no purge gap is needed because no row depends on another period |
| `models/train_baseline.py` | CLI: naive majority baseline + `StandardScaler`+`LogisticRegression`, C chosen on validation only, refit on train+validation, one held-out test evaluation; metrics, standardised coefficients, eight pass/fail checks |
| `models/train_news_aware.py` | CLI: same model family on market-only and market+news features over the identical split, C grid and test rows; comparison report, joint predictions, ten pass/fail checks |
| `trading/simulator.py` | Deterministic long-only portfolio engine (fills at the open, marks at the close, flat-bps costs, no overspending), buy-and-hold and 5-session momentum action generators, and the Phase 5 benchmark artifact |

Artifact contract:

* `reports/experiment_results/phase1_data_report.json` — config snapshot, per-table quality
  reports, coverage, ten explicit pass/fail checks, columns still unassigned, known limitations.
* `reports/experiment_results/phase2_classification_report.json` — method (taxonomy version, text
  scope, sentiment mode), category distribution with an explicit imbalance block, evidence-flag
  counts, worked examples per category, and the limitations carried forward.
* `reports/experiment_results/phase2_annotation_sheets.json` — the review sheets, their sampling
  method, the allowed label values, and the statement that no labels were written. Labelling adds
  `data/annotations/<sheet>_labelling_log.jsonl` (append-only: timestamp, article, old label, new
  label, reviewer), so a reviewed sheet is always reconstructible.
* `reports/experiment_results/phase2_classifier_evaluation.json` — metrics against human labels
  (or an explicit "no labels yet" record), plus `phase2_misclassifications.csv`.
* `reports/experiment_results/phase3_baseline_report.json` — config snapshot, the cutoff and target
  rules as text, feature definitions, the warm-up drop count, split ranges with dates and class
  balance, naive and market-only metrics on validation and test, the C grid, standardised
  coefficients, eight pass/fail checks, and the limitations.
* `reports/experiment_results/phase3_test_predictions.csv` — one row per held-out session
  (`session_date, decision_time_utc, next_session_return, target, probability_up, prediction,
  correct`), so every reported test number traces to a date and a probability.
* `reports/experiment_results/phase4_news_comparison_report.json` — config snapshot, cutoff and
  news-window rules, market and news feature definitions, the `no_news` share, split ranges with
  dates and class balance, market-only and market-plus-news metrics, both selected C values,
  standardised coefficients (all features), ten pass/fail checks, comparison deltas and limitations.
* `reports/experiment_results/phase4_test_predictions.csv` — one row per held-out session with both
  models' probabilities and predictions, so every comparison number traces to a date.
* `data/processed/features_<SYMBOL>_<interval>.csv` — the market-only feature table, recorded in
  `data/manifest.json`.
* `data/processed/features_<SYMBOL>_<interval>_news.csv` — the combined market + news feature
  store, also recorded in `data/manifest.json` with its row/column contract.
* `reports/experiment_results/phase5_simulation_report.json` — capital, sizing, execution and cost
  rules as text, the period, per-strategy return/final value/trades/costs/max drawdown, thirteen
  pass/fail checks, and limitations.
* `reports/experiment_results/phase5_portfolio_values.csv` — one equity row per strategy per
  session (`strategy, session_date, action, executed, shares, cash, portfolio_value, ...`).
* `reports/experiment_results/phase6_rl_report.json` — PPO protocol, validation-only candidate
  selection, per-seed held-out metrics, benchmark comparison, 15 checks and limitations.
* `reports/experiment_results/phase6_test_portfolio_values.csv` — one held-out equity row per
  PPO seed and benchmark per session, consumed directly by `/simulation` and the dashboard.

Paths inside every artifact are project-relative, so reports are diffable and portable. The
backend/dashboard and later phases read these files — nothing re-derives numbers by hand.

Implemented through Phase 7: `trading/gym_env.py`, `trading/train_rl.py`, the Phase 6 experiment artifacts, `src/stock_project/backend/`, and `frontend/index.html`. The browser loads pinned React and Plotly builds from jsDelivr; API reads and all automated tests remain offline.
