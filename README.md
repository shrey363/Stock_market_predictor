# Geopolitical Event-Aware Stock Market Prediction and Trading

> **AI/ML Mini Project · FinTech · Team of 2**
> **Status:** In development — see [docs/ROADMAP.md](docs/ROADMAP.md) for live progress and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the target design.
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
├── PROJECT_DESCRIPTION.md  # Full original specification
├── .gitignore
├── requirements.txt
├── configs/
│   └── config.example.yaml
├── data/
│   ├── raw/                # Downloaded/original data; not committed
│   ├── interim/            # Intermediate cleaned datasets
│   └── processed/          # Features ready for experiments
├── notebooks/              # Exploration and baseline experiments
├── src/stock_project/
│   ├── data/               # market_data.py, news_data.py
│   ├── preprocessing/      # cleaning scripts
│   ├── nlp/                # event_classifier.py, sentiment.py
│   ├── features/           # technical_indicators.py, news_features.py
│   ├── models/             # train_baseline.py, train_news_aware.py, evaluate.py
│   ├── trading/            # simulator.py, gym_env.py, train_rl.py
│   └── common/
├── backend/app.py          # Flask API
├── frontend/               # React dashboard
├── tests/                  # Data, feature, and simulator tests
├── reports/                # Figures and experiment results
└── docs/
    ├── ARCHITECTURE.md     # Target system design (data flow, layers, invariants)
    ├── ROADMAP.md          # Live progress log — updated with every change
    ├── dataset_notes.md    # (planned)
    └── model_evaluation.md # (planned)
```

*(Most of this structure does not exist yet — it is the target layout, created lazily as stages need it.)*

## Development phases

Full detail, checklists, and the running change log live in **[docs/ROADMAP.md](docs/ROADMAP.md)**; the target system design lives in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

| Phase | Goal | Deliverable |
|---|---|---|
| 1 | Define and inspect data | Documented, reproducible sample datasets |
| 2 | News classification and sentiment | News dataset with traceable category/sentiment outputs |
| 3 | Market features and baseline prediction | Reproducible baseline metrics |
| 4 | Add geopolitical features | Fair market-only vs market+news comparison |
| 5 | Trading simulator | Tested historical simulator |
| 6 | Reinforcement learning | RL evaluation report with metrics and limitations |
| 7 | Dashboard and final integration | Working demo and final report |

## Getting started

```bash
# 1. Clone and enter the repository
git clone <repo-url>
cd geopolitical-stock-predictor

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies (created in Phase 1)
pip install -r requirements.txt
```

Concrete commands for running each stage will be added to this README as those stages land (tracked in the roadmap).

## Testing

```bash
pytest
```

The specification requires, at minimum: date/timezone parsing checks, duplicate detection, missing-data reporting, leakage guards on features and cutoffs, chronological split verification, simulator cash/position/cost correctness, and agreement between dashboard values and saved experiment results. Tests appear alongside the code they cover.

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
