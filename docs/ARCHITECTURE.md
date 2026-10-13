# Architecture

Basic system architecture for the **Geopolitical Event-Aware Stock Market Prediction and Trading** project. This describes the *target* design; components are implemented phase by phase per [ROADMAP.md](ROADMAP.md). No application code exists yet beyond documentation.

- Specification: [PROJECT_DESCRIPTION.md](../PROJECT_DESCRIPTION.md)
- Overview and setup: [README.md](../README.md)

---

## 1. High-level data flow

```text
                        ┌──────────────────────────────────────────────────────────┐
                        │                        DATA LAYER                       │
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
                 ┌──────────────▼──────────────┐    ┌───────────▼────────────────┐
                 │   SUPERVISED MODELS         │    │   TRADING SIMULATOR        │
                 │  train_baseline.py          │    │  simulator.py (deterministc)│
                 │   (naive + market-only)     │    │  gym_env.py (Gymnasium)     │
                 │  train_news_aware.py        │    │  train_rl.py (SB3 agent)    │
                 │  evaluate.py (chronological)│    │  benchmarks: buy&hold, rule │
                 └──────────────┬──────────────┘    └───────────┬────────────────┘
                                │                               │
                                │        metrics.json /         │
                                │        results CSV/JSON       │
                 ┌──────────────▼───────────────────────────────▼──────────────┐
                 │                        API LAYER (Flask)                    │
                 │   endpoints: /market, /news, /predictions, /models,        │
                 │               /simulation, /methodology                    │
                 └──────────────────────────────┬──────────────────────────────┘
                                                │ JSON
                 ┌──────────────────────────────▼──────────────────────────────┐
                 │                  FRONTEND (React + Plotly)                  │
                 │  Market overview · News explorer · Prediction view ·        │
                 │  Model comparison · Trading simulation · Limitations       │
                 └─────────────────────────────────────────────────────────────┘
```

## 2. Layers and responsibilities

| Layer | Modules (target) | Responsibility |
|---|---|---|
| **Ingestion** | `src/stock_project/data/market_data.py`, `news_data.py` | Fetch/load raw OHLCV and news; record source and retrieval date; never modify raw files |
| **Preprocessing** | `preprocessing/clean_market_data.py`, `clean_news_data.py` | Parse dates and timezones, deduplicate, report missing values, clean text |
| **NLP** | `nlp/event_classifier.py`, `nlp/sentiment.py` | Assign event category (rule baseline first) and sentiment label/score; keep outputs traceable to article_id |
| **Features** | `features/technical_indicators.py`, `features/news_features.py` | Session-level market indicators; news aggregation using only pre-cutoff articles; explicit no-news indicator |
| **Models** | `models/train_baseline.py`, `train_news_aware.py`, `evaluate.py` | Naive baseline, market-only and market+news models with matched settings; chronological evaluation |
| **Trading** | `trading/simulator.py`, `gym_env.py`, `train_rl.py` | Deterministic portfolio simulator, Gymnasium wrapper, SB3 training, benchmark strategies |
| **Backend** | `backend/app.py` (Flask) | Read-only JSON API over saved datasets and experiment results |
| **Frontend** | `frontend/` (React + Plotly) | Dashboard sections per §11 of the specification |
| **Common** | `common/` | Config loading, paths, logging, seeds |

## 3. Key design rules

These are architectural invariants, not suggestions — a component that breaks them is wrong regardless of its metrics.

1. **Chronological everything.** Train/validation/test splits are time-ordered; no random shuffling of series. Fitted transformations are fit on training data only.
2. **Prediction cutoff.** Every feature row is built from information available strictly before the session's prediction time. News published after the cutoff is excluded, not down-weighted.
3. **No silent fills.** Missing news → zero counts *plus* a missing-news indicator. Never impute neutral sentiment for absence of data.
4. **Traceability.** Every classified headline keeps its `article_id`, original text, timestamp, source, assigned category, and sentiment score so dashboard values can be traced to inputs.
5. **Simulator determinism.** Given the same prices, costs, and actions, the simulator produces identical results; it is unit-tested with hand-checkable examples before RL touches it.
6. **RL information set.** The Gymnasium observation contains only data available at the simulated time (no future prices, news, or the held-out split).
7. **Experiment artifacts are files.** Models, metrics, and simulation runs are written to `reports/experiment_results/` as reproducible artifacts; the backend and dashboard only *read* them, so displayed numbers always match saved results.

## 4. Data contracts

Full field lists are in §6 of the specification. Summary of the two core tables:

- **Market table:** `symbol, date, open, high, low, close, adj_close, volume, source`
- **News table:** `article_id, published_at, headline, article_text, source, event_category, sentiment_label, sentiment_score, affected_sector`
- **Feature table (processed):** one row per (symbol, session) — lagged returns, MA/RSI/volatility, per-category article counts, mean sentiment, pos/neg/neutral counts, hours since last relevant article, `no_news` indicator, and the binary target.

Storage stays CSV/JSON on disk until a real need arises; MongoDB is explicitly a later option, not a default dependency.

## 5. API surface (Planned, Phase 7)

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
