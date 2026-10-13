# Roadmap

Living progress tracker for the **Geopolitical Event-Aware Stock Market Prediction and Trading** project.

- **Convention:** every phase has a checklist. A box is ticked only when the work is done *and* verified.
- **Change log rule:** every meaningful action taken on this project gets an entry under [Change Log](#change-log) — newest first. Nothing counts as done until it is logged here.
- Full requirements, schemas, and risk register: [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md).
- Setup and overview: [README.md](README.md).

**Overall status:** Phase 1 — Define and inspect data (not started)

---

## Phase 1 — Define and inspect data

**Goal:** documented, reproducible sample datasets.

- [ ] Choose one market/index or a small set of stocks.
- [ ] Select a small historical geopolitical-news dataset.
- [ ] Record licences, data sources, columns, date coverage, and known limitations.
- [ ] Inspect missing values, duplicates, timestamps, and class balance.
- [ ] Load both datasets in Python and inspect rows, date ranges, missing values, duplicates.
- [ ] Set up repository scaffolding: `.gitignore`, `requirements.txt`, `configs/config.example.yaml`, `data/` folders.

## Phase 2 — News classification and sentiment

**Goal:** a news dataset with traceable category/sentiment outputs.

- [ ] Define event categories and write annotation guidelines with examples.
- [ ] Create or verify a small manually reviewed labelled sample.
- [ ] Implement a transparent keyword/rule-based event-classification baseline.
- [ ] Evaluate the baseline on the reviewed sample; report class imbalance.
- [ ] Add sentiment analysis (simple baseline or pretrained financial model, e.g. FinBERT).
- [ ] Review misclassifications and document limitations.

## Phase 3 — Market features and baseline prediction

**Goal:** reproducible baseline metrics.

- [ ] Clean market data; compute returns, moving averages, RSI, volatility.
- [ ] Define the prediction cutoff and the next-session binary target.
- [ ] Train a naive/majority baseline and a market-only model (Logistic Regression or Random Forest).
- [ ] Evaluate with chronological train/validation/test splits — no shuffling.
- [ ] Report accuracy, precision, recall, F1, balanced accuracy against the naive baseline.
- [ ] Leakage checks: no future prices, future news, or revised data in features.

## Phase 4 — Add geopolitical features

**Goal:** a fair comparison answering the project's core research question.

- [ ] Aggregate news into per-session features using only pre-cutoff articles (category counts, mean sentiment, pos/neg/neutral counts, time since last article, no-news indicator).
- [ ] Train a market-plus-news model with the same family and comparable settings as Phase 3.
- [ ] Compare both models on the same held-out test period.
- [ ] Analyse errors and feature limitations; document what did not work.

## Phase 5 — Trading simulator

**Goal:** a tested historical simulator.

- [ ] Define initial capital, position limits, execution timing, and transaction costs.
- [ ] Implement deterministic buy/sell/hold mechanics with explicit action semantics.
- [ ] Unit-test cash, holdings, execution, and portfolio-value calculations with hand-checkable examples.
- [ ] Implement buy-and-hold and a simple rule-based benchmark strategy.
- [ ] Verify the simulator never spends more cash than available and applies costs consistently.

## Phase 6 — Reinforcement learning

**Goal:** RL evaluation report with metrics and limitations.

- [ ] Define observations (only information available at simulated time), actions, and reward.
- [ ] Wrap the tested simulator in a Gymnasium-compatible environment.
- [ ] Train a small agent with Stable-Baselines3.
- [ ] Tune only on training/validation periods.
- [ ] Evaluate on a separate chronological test period, across multiple seeds where practical.
- [ ] Report cumulative return, Sharpe ratio (documented frequency), max drawdown, trade count, costs — against buy-and-hold and the rule-based strategy.

## Phase 7 — Dashboard and final integration

**Goal:** working demo and final project report.

- [ ] Expose data, predictions, comparisons, and simulation results through Flask.
- [ ] Build the React + Plotly dashboard (or agreed alternative).
- [ ] Sections: market overview, news explorer, prediction view, model comparison, trading simulation, methodology & limitations.
- [ ] Test the complete workflow from input data to displayed results; verify dashboard values match saved experiment outputs.
- [ ] Document setup, reproduction commands, and limitations in the README.
- [ ] Compile the final report including failed experiments and honest caveats.

---

## Definition of Done

Mirrors §19 of the specification — the project is complete when:

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

## Change Log

Newest first. Every substantive action gets an entry: date, what changed, files touched, and verification.

### 2026-10-13

- **Initialized documentation.** Read `PROJECT_DESCRIPTION.md` and authored:
  - `README.md` — overview, principles, event taxonomy, tech stack, target repo structure, phase summary table, setup and testing instructions, limitations.
  - `ROADMAP.md` (this file) — seven-phase checklist derived from §14, Definition of Done from §19, and this change log.
- **Verified:** both files cross-check against the specification's milestones and non-goals; repo contained only `PROJECT_DESCRIPTION.md` before this change.
- **Not yet started:** any Phase 1 checklist item (no code, data, or dependencies exist yet).
