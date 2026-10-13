# Geopolitical Event-Aware Stock Market Prediction and Trading

> **Project type:** AI/ML Mini Project  
> **Domain:** Financial Technology (FinTech)  
> **Team size:** 2  
> **Status:** Planned — build incrementally, starting with small historical datasets  
> **Important:** This project is for research and simulated trading only. It does not provide financial advice or execute real trades.

## 1. Project Overview

Stock-market movements are influenced by historical price patterns, economic conditions, investor sentiment, and geopolitical developments. Events such as armed conflicts, trade restrictions, sanctions, diplomatic decisions, and disruptions to supply chains can affect market expectations and different industry sectors.

This project aims to build a system that combines historical stock-market data with geopolitical-news analysis. Natural Language Processing (NLP) will extract useful information from news, supervised Machine Learning (ML) will predict market movement, and a Reinforcement Learning (RL) agent will learn buy, sell, or hold actions inside a simulated trading environment.

The system will present news, model predictions, and simulated trading performance in a dashboard. Development will proceed in small, testable stages. The initial version will use a small historical dataset and a limited set of event categories; the dataset, classification pipeline, prediction model, and simulator will be expanded only after the earlier stages work reliably.

## 2. Problem Statement

Many basic stock-prediction systems primarily use historical prices and technical indicators. Such systems may not represent potentially relevant information contained in geopolitical news. This project investigates whether adding structured geopolitical-news features to historical market data improves the prediction of subsequent stock-market movement, and whether those predictions can support a simulated trading strategy.

The project does **not** assume that geopolitical news always moves markets or that a model can reliably predict market prices. It will test these ideas against chronological historical data and compare results with simple baselines.

## 3. Goals and Objectives

1. Collect historical market data and dated geopolitical-news data.
2. Clean and align market and news data by date/time.
3. Classify news into a manageable set of geopolitical event categories.
4. Extract sentiment and other useful features from news text.
5. Calculate technical indicators from market data.
6. Train a baseline model using market data alone.
7. Train a second model using market data plus geopolitical-news features.
8. Compare the models on unseen, later historical data.
9. Implement a Reinforcement Learning agent that chooses buy, sell, or hold actions in a simulated trading environment.
10. Compare the RL agent against simple trading baselines using consistent assumptions.
11. Display data, predictions, and simulation results in an interactive dashboard.
12. Document limitations, failed experiments, and evaluation results honestly.

## 4. Scope and Non-Goals

### In scope

- Small, historical datasets for the first prototype.
- A limited initial list of event categories.
- Text preprocessing and news classification.
- Financial sentiment analysis using a pretrained model or a simpler baseline.
- Technical indicators and supervised market-movement prediction.
- A historical trading simulator with buy/sell/hold actions.
- RL training and evaluation after the simulator has been tested.
- A dashboard for analysis and visualisation.
- Reproducible experiments and documented metrics.

### Out of scope for the initial version

- Live or real-money trading.
- Claims of guaranteed profit or reliable price prediction.
- Covering every stock, country, language, or geopolitical event.
- Training a large language model from scratch.
- Building a complex, production-grade financial platform.
- Assuming that correlation proves that a geopolitical event caused a market movement.

## 5. Intended Workflow

The planned data flow is:

1. **Market-data collection** — obtain dated OHLCV data (open, high, low, close, volume) for a small number of instruments.
2. **News-data collection** — obtain historical geopolitical headlines/articles with publication timestamps and, where possible, source information.
3. **Preprocessing** — normalise dates, handle missing values, remove duplicate articles, clean text, and check data quality.
4. **Event classification** — assign each article an event category. Start with transparent keyword/rule-based labels or an existing labelled dataset; consider a trained classifier only when enough labelled examples are available.
5. **Sentiment analysis** — estimate positive, negative, or neutral financial/news sentiment using a simple baseline or a pretrained financial NLP model such as FinBERT. Sentiment is not the same as the event's severity or its impact on a specific stock.
6. **Feature aggregation** — convert articles into daily or session-based features, such as counts by category, average sentiment, negative/positive article counts, and recency. Only use news that was available before the prediction cutoff.
7. **Market feature engineering** — calculate returns, moving averages, RSI, volatility, and other indicators using information available at the prediction time.
8. **Supervised prediction** — predict a clearly defined target, initially whether the next trading session's return is positive or negative.
9. **Comparison and evaluation** — compare a market-only baseline with a market-plus-news model on a chronological test period.
10. **Trading simulation** — use historical prices and model features to simulate portfolio holdings, cash, orders, and trading costs.
11. **Reinforcement Learning** — train an agent in the tested simulator to choose buy, sell, or hold actions.
12. **Dashboard** — show market data, news classifications, model predictions, model comparisons, and simulated portfolio performance.

## 6. Initial Data Strategy

Begin with a small, manageable dataset rather than attempting to cover the entire market.

### Market data

- Start with one index or a small set of stocks.
- Use a historical source such as Yahoo Finance through `yfinance`, subject to its availability and applicable terms.
- Store the instrument symbol, trading date, open, high, low, close, adjusted close (when available), and volume.
- Record the source and retrieval date in the dataset documentation.

### News data

- Start with a small historical dataset containing news headlines and timestamps.
- A dataset from Kaggle or another documented research source may be used after checking its licence, schema, dates, and suitability.
- A news API can be considered later if historical coverage and access terms meet the project's needs.
- Keep the original headline/text and timestamp so that predictions can be traced back to their inputs.

### Suggested initial schema

**Market table/file**

| Field | Meaning |
|---|---|
| `symbol` | Stock or index identifier |
| `timestamp` / `date` | Market session date/time |
| `open` | Opening price |
| `high` | Highest price |
| `low` | Lowest price |
| `close` | Closing price |
| `adj_close` | Adjusted close, if available |
| `volume` | Trading volume |
| `source` | Data source |

**News table/file**

| Field | Meaning |
|---|---|
| `article_id` | Unique article identifier |
| `published_at` | Publication timestamp, including timezone if known |
| `headline` | Original headline |
| `article_text` | Article body, if available and permitted |
| `source` | Publisher or dataset source |
| `event_category` | Assigned category; may initially be unknown |
| `sentiment_label` | Positive, negative, or neutral |
| `sentiment_score` | Model score, if available |
| `affected_sector` | Optional sector label, initially manual/rule-based if needed |

The actual schema can be simplified to match the dataset chosen. Do not invent missing timestamps, labels, or sector associations.

## 7. Initial Event Categories

Start with a small and clearly defined taxonomy. For example:

- `armed_conflict`
- `trade_restriction`
- `sanctions`
- `diplomatic_development`
- `energy_or_supply_disruption`
- `other_or_uncertain`

These are proposed project labels, not categories already present in any dataset. Define simple annotation rules and examples for each category before labelling data. Keep an `other_or_uncertain` option rather than forcing every headline into a confident category.

### Classification approach

1. Inspect the dataset and count available examples.
2. Create a small labelled sample and a written labelling guide.
3. Establish a transparent keyword/rule-based baseline.
4. Evaluate it on a manually reviewed sample.
5. If sufficient labelled data is available, train and compare a text classifier.
6. Review misclassifications and report class imbalance.

Do not treat a keyword match as proof that an event happened. Negation, speculation, old events, and headlines discussing several countries can make classification difficult. The first version may classify only headlines and should document this limitation.

## 8. Sentiment and Feature Engineering

Sentiment and event classification are separate tasks:

- **Event category:** what kind of event the text describes.
- **Sentiment:** the tone expressed in the text.
- **Severity:** how consequential the event may be, which needs a defined rubric or reliable labels.
- **Market impact:** the observed or predicted response of a particular market/sector; this must be evaluated rather than inferred from sentiment alone.

For each market session, aggregate only news published before the chosen prediction cutoff. Candidate features include:

- Number of articles in each event category.
- Mean or median sentiment score.
- Counts of positive, negative, and neutral articles.
- Number of high-priority events, only if severity is defined consistently.
- Time since the most recent relevant article.
- Technical indicators and lagged market returns.

If no relevant article exists for a session, represent that situation explicitly (for example, zero article count plus a missing-news indicator). Do not silently treat missing data as neutral sentiment.

## 9. Prediction Target and Models

### Initial target

Begin with a binary target:

- `1`: next trading session's return is greater than zero.
- `0`: next trading session's return is zero or below.

The team may revise the definition if the selected data or research question requires it, but it must be documented and kept consistent across models.

### Models

Start simple:

- **Baseline:** a majority-class or other clearly stated naive baseline.
- **Market-only model:** Logistic Regression or Random Forest using market features.
- **Market + geopolitical-news model:** the same model family and comparable settings, with news features added.

More complex models should be considered only if the baseline experiment is sound and there is a clear reason to add them. Avoid starting with a deep neural network simply because it is available.

### Evaluation

- Split data chronologically into train, validation, and test periods.
- Do not randomly shuffle time-series observations.
- Fit preprocessing and feature transformations on training data only.
- Avoid using future prices, future news, or revised information that would not have been available at prediction time.
- Report accuracy alongside precision, recall, F1-score, and balanced accuracy where useful.
- Inspect class balance and compare against naive baselines.
- Preserve a final test period for evaluation after model selection.

The goal is to test whether news features add useful predictive information, not to guarantee that they will improve results.

## 10. Reinforcement Learning Trading Simulator

RL is a planned project component, but it should be implemented only after the market-data pipeline, prediction experiments, and simulator mechanics have been tested.

### Environment definition

**State/observation:** a compact set of information available at the current simulated time, such as recent returns, technical indicators, news features, the model's predicted probability, current cash, and current holdings.

**Actions:** begin with a small action space such as:
- `hold`
- `buy`
- `sell`

Specify whether actions trade a fixed quantity, a fixed fraction of available cash, or a target portfolio position. Avoid ambiguous action semantics.

**Reward:** start with a documented reward based on the change in portfolio value or net portfolio return, after transaction costs. Add penalties only when they have a clear rationale. Do not design the reward using future information.

**Environment:** step through historical sessions in chronological order. The simulator must define initial capital, order execution timing, position limits, transaction costs, and what happens when an action cannot be executed.

### Suggested implementation sequence

1. Implement a simple deterministic simulator first.
2. Test cash, holdings, order execution, and portfolio-value calculations with hand-checkable examples.
3. Implement and evaluate buy-and-hold and a simple rule-based strategy.
4. Expose the simulator through a Gymnasium-compatible environment.
5. Train a small RL agent using Stable-Baselines3, selecting an algorithm appropriate to the action space.
6. Tune only on training/validation periods.
7. Evaluate on a separate chronological test period.
8. Repeat across random seeds where practical and document variability.

### RL evaluation

Compare the agent with the same baselines using the same test period, starting capital, execution rules, and cost assumptions. Report:

- Cumulative and total portfolio return.
- Sharpe ratio, with the calculation and return frequency documented.
- Maximum drawdown.
- Number of trades and transaction costs.
- Comparison with buy-and-hold and any simple rule-based strategy.

Avoid claiming that the agent has learned a profitable general strategy based on a single historical period. RL agents can overfit historical paths and reward-function choices. Simulation results do not establish future profitability.

## 11. Dashboard

The dashboard should make the pipeline and its limitations easy to inspect. Suggested sections:

1. **Market overview:** instrument, price chart, returns, and selected indicators.
2. **News explorer:** headline, publication time, event category, sentiment, and source.
3. **Prediction view:** predicted direction/probability and the features used.
4. **Model comparison:** market-only versus market-plus-news metrics.
5. **Trading simulation:** portfolio value, holdings, trades, returns, drawdown, and benchmark comparison.
6. **Methodology and limitations:** data sources, prediction cutoff, evaluation period, and important caveats.

Predictions should be presented as model outputs, not as financial advice.

## 12. Suggested Technology Stack

Use a small stack first and add dependencies only when required.

| Layer | Initial choice | Purpose |
|---|---|---|
| Language | Python | Main project language |
| Data handling | Pandas, NumPy | Data cleaning and feature engineering |
| Market data | `yfinance` or a documented historical dataset | Historical OHLCV data |
| Baseline ML | Scikit-learn | Classification and evaluation |
| NLP | Hugging Face Transformers; optionally FinBERT | Sentiment analysis or text classification |
| RL environment | Gymnasium | Define the trading environment |
| RL algorithms | Stable-Baselines3 | Train and evaluate an RL agent |
| Backend | Flask | API endpoints for data and model results |
| Frontend | React, JavaScript, HTML, CSS | Interactive user interface |
| Charts | Plotly | Market and portfolio visualisation |
| Storage | CSV/JSON initially; MongoDB if needed | Store datasets, metadata, and results |

The original proposal includes React and Streamlit. To avoid maintaining two frontends, the recommended initial plan is React + Flask + Plotly. Streamlit can be used instead for a faster prototype if the team decides not to build a separate React frontend.

## 13. Proposed Repository Structure

Adjust this structure as the implementation develops. Do not create empty complexity before it is needed.

```text
geopolitical-stock-predictor/
├── README.md
├── PROJECT_DESCRIPTION.md
├── .gitignore
├── requirements.txt
├── configs/
│   └── config.example.yaml
├── data/
│   ├── README.md
│   ├── raw/                 # Downloaded/original data; normally not committed
│   ├── interim/             # Intermediate cleaned datasets
│   └── processed/           # Features ready for experiments
├── notebooks/
│   ├── 01_explore_market_data.ipynb
│   ├── 02_explore_news_data.ipynb
│   └── 03_baseline_experiments.ipynb
├── src/
│   └── stock_project/
│       ├── data/
│       │   ├── market_data.py
│       │   └── news_data.py
│       ├── preprocessing/
│       │   ├── clean_market_data.py
│       │   └── clean_news_data.py
│       ├── nlp/
│       │   ├── event_classifier.py
│       │   └── sentiment.py
│       ├── features/
│       │   ├── technical_indicators.py
│       │   └── news_features.py
│       ├── models/
│       │   ├── train_baseline.py
│       │   ├── train_news_aware.py
│       │   └── evaluate.py
│       ├── trading/
│       │   ├── simulator.py
│       │   ├── gym_env.py
│       │   └── train_rl.py
│       └── common/
├── backend/
│   └── app.py
├── frontend/
│   └── ...
├── tests/
│   ├── test_data_validation.py
│   ├── test_features.py
│   └── test_trading_simulator.py
├── reports/
│   ├── figures/
│   └── experiment_results/
└── docs/
    ├── dataset_notes.md
    └── model_evaluation.md
```

## 14. Development Milestones

### Phase 1 — Define and inspect data
- [ ] Choose one market/index or a small set of stocks.
- [ ] Select a small historical news dataset.
- [ ] Record licences, data sources, columns, date coverage, and known limitations.
- [ ] Inspect missing values, duplicates, timestamps, and class balance.
- **Deliverable:** documented, reproducible sample datasets.

### Phase 2 — News classification and sentiment
- [ ] Define event categories and annotation guidelines.
- [ ] Create or verify a small manually reviewed labelled sample.
- [ ] Implement a simple event-classification baseline.
- [ ] Add sentiment analysis.
- [ ] Review errors and document limitations.
- **Deliverable:** a news dataset with traceable category/sentiment outputs.

### Phase 3 — Market features and baseline prediction
- [ ] Clean market data and calculate returns/indicators.
- [ ] Define the prediction cutoff and next-session target.
- [ ] Train a naive baseline and market-only model.
- [ ] Evaluate using chronological splits.
- **Deliverable:** reproducible baseline metrics.

### Phase 4 — Add geopolitical features
- [ ] Aggregate news into market-session features without future leakage.
- [ ] Train a market-plus-news model.
- [ ] Compare it with the market-only model on the same test period.
- [ ] Analyse errors and feature limitations.
- **Deliverable:** a fair comparison answering the project's core research question.

### Phase 5 — Trading simulator
- [ ] Define capital, positions, execution timing, and transaction costs.
- [ ] Implement and unit-test deterministic trading mechanics.
- [ ] Add buy-and-hold and a simple rule-based benchmark.
- **Deliverable:** a tested historical simulator.

### Phase 6 — Reinforcement Learning
- [ ] Define observations, actions, and rewards.
- [ ] Wrap the simulator in a Gymnasium-compatible environment.
- [ ] Train a small RL agent.
- [ ] Tune on training/validation data only.
- [ ] Compare against baselines on held-out historical data.
- **Deliverable:** RL evaluation report with metrics and limitations.

### Phase 7 — Dashboard and final integration
- [ ] Expose the required results through Flask.
- [ ] Build the React dashboard or use the agreed alternative.
- [ ] Connect charts, news, predictions, and simulation outputs.
- [ ] Test the complete workflow from input data to displayed results.
- [ ] Document setup, reproduction steps, and limitations.
- **Deliverable:** working demo and final project report.

## 15. Testing and Quality Checks

At minimum, test the following:

- Dates parse correctly and timezones are handled consistently.
- Duplicate articles and duplicate market rows are detected.
- Missing data is reported rather than silently hidden.
- Feature calculations do not use future observations.
- News published after the prediction cutoff is excluded.
- Train/test splitting preserves chronology.
- Model metrics are computed on the intended split.
- The simulator does not allow spending more cash than available.
- Positions and portfolio value update correctly after trades.
- Transaction costs are applied consistently.
- RL observations contain only information available at that time.
- Dashboard values match saved experiment results.

## 16. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Too little labelled news | Begin with few classes, inspect class counts, and label a small evaluation sample carefully |
| Event categories overlap | Write category definitions and permit `other_or_uncertain` |
| News timestamp ambiguity | Record timezone assumptions and use conservative cutoff rules |
| Look-ahead/data leakage | Use chronological splits and enforce feature cutoffs |
| Imbalanced target classes | Report class balance and use suitable metrics/baselines |
| Unstable RL performance | Use simple rewards, baselines, multiple seeds where feasible, and a held-out period |
| Dataset/API restrictions | Check licences and access terms before use |
| Scope creep | Complete each milestone and its deliverable before expanding |

## 17. Expected Outcomes

By the end of the project, the team aims to deliver:

- A documented historical market/news dataset pipeline.
- A geopolitical-news classification and sentiment-analysis component.
- A supervised stock-movement prediction baseline.
- A comparison between market-only and market-plus-news models.
- A tested historical trading simulator.
- An RL agent that can be evaluated in that simulator.
- A dashboard for exploring news, predictions, and simulated performance.
- A report containing evaluation metrics, limitations, and reproducible experiments.

These are intended deliverables, not guaranteed model-performance outcomes.

## 18. Limitations and Responsible Use

- Historical patterns may not persist in future markets.
- News sources can be incomplete, delayed, biased, or incorrect.
- Sentiment does not directly determine the direction of a stock's return.
- Geopolitical events can affect sectors differently and may already be reflected in prices.
- Small datasets may not represent rare or unprecedented events.
- Backtesting and RL can overfit historical data.
- Transaction costs, liquidity, slippage, and execution timing can materially change results.
- The system is an educational research prototype, not a financial-advice service or a guarantee of future returns.

## 19. Definition of Done

The project is considered complete when:

- [ ] Data sources and preparation steps are documented.
- [ ] Event labels and sentiment outputs can be inspected.
- [ ] Market-only and market-plus-news models are evaluated fairly.
- [ ] Leakage checks and chronological splitting are documented.
- [ ] The trading simulator passes basic correctness tests.
- [ ] The RL agent is compared against simple baselines on held-out data.
- [ ] Dashboard results agree with experiment outputs.
- [ ] The repository contains setup instructions and reproducible commands.
- [ ] Limitations and unsuccessful experiments are included in the final report.

---

## 20. First Tasks to Start With

Do these in order; do not begin by training the RL agent.

1. Create the repository and this project description.
2. Choose one market/index or a small set of stocks.
3. Find and inspect a small historical geopolitical-news dataset.
4. Write down the dataset schema, time coverage, licence, and label availability.
5. Load both datasets in Python and inspect their rows, date ranges, missing values, and duplicates.
6. Decide the first event categories and manually review a small sample of headlines.
7. Produce a first simple event-classification output before adding advanced models.

**Guiding principle:** Build a small, correct, measurable end-to-end system first. Expand the dataset, models, RL environment, and dashboard only after the previous stage has been tested.
