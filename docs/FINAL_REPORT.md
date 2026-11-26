# Final Project Report

## Geopolitical Event-Aware Stock Market Prediction and Trading

This research prototype tests whether structured geopolitical-news features improve next-session SPY direction prediction and whether a PPO agent can turn the same historical information into a useful simulated trading policy. It is an educational, historical simulation only—not financial advice and not evidence of future profitability.

## Data and evaluation contract

- **Market:** 1,259 SPY daily sessions, 2018-01-02 through 2022-12-30.
- **News:** 5,930 HuffPost Politics articles from the pinned `heegyu/news-category-dataset` revision, licensed CC BY 4.0.
- **Features:** 1,209 sessions after a 50-session indicator warm-up; 10 market and 13 news features.
- **Split:** chronological 846 train / 181 validation / 182 test sessions. Nothing is shuffled.
- **Cutoff:** market features for session `t` use bars through `t-1`; day-precision news becomes available the following day and must be available by the session cutoff.
- **Trading:** simulated only, long-only, whole shares, fills at the open, marks at the close, and 10 bps transaction cost on buys and sells.

## Results

### News classification

The rule classifier scored **90.5% accuracy**, close to the 90.0% naive-majority baseline, but only **42.3% balanced accuracy**, macro F1 0.468, and Cohen's kappa 0.270 on 200 human-labelled articles. It is primarily an `other_or_uncertain` detector and is weak on rare geopolitical categories. Those machine labels therefore limit every downstream news feature.

### Supervised movement prediction

On the same 182 held-out sessions:

| Model | Accuracy | Balanced accuracy | ROC-AUC |
|---|---:|---:|---:|
| Naive majority | 0.429 | 0.500 | — |
| Market only | 0.445 | 0.506 | 0.532 |
| Market + news | 0.456 | 0.514 | 0.525 |

News raises balanced accuracy by only **0.008** while ROC-AUC falls by 0.007. With one instrument and 182 test sessions, this is indistinguishable from noise and does not demonstrate predictive skill.

### Deterministic simulator

Across the complete 2018–2022 history, buy-and-hold returned **+42.53%** with one trade. The five-session momentum rule returned **−33.13%** across 323 trades and paid 31,227.96 in transaction costs. This failed rule is useful evidence: turnover and costs can overwhelm a simple historical signal.

### Reinforcement learning

PPO hyperparameters were selected on validation only, then final agents were fit on train+validation and evaluated once per seed on the held-out test period.

| Strategy | Held-out return | Sharpe | Trades | Max drawdown |
|---|---:|---:|---:|---:|
| PPO seed 0 | +0.84% | 0.160 | 29 | 14.85% |
| PPO seed 1 | −8.14% | −0.555 | 35 | 14.64% |
| PPO seed 2 | −7.54% | −0.325 | 13 | 19.60% |
| Buy and hold | −13.75% | −0.677 | 1 | 19.84% |
| Momentum 5 | −25.03% | −2.179 | 57 | 25.30% |
| Cash | 0.00% | — | 0 | 0.00% |

PPO averaged **−4.94%** with a 5.02 percentage-point sample standard deviation. Every seed beat buy-and-hold during this falling test period, but only one beat cash. The agent is seed-sensitive and did not learn a reliably profitable policy.

## Final integration

A loopback-only Flask application reads the saved Phase 1–6 CSV/JSON artifacts and exposes `/market`, `/news`, `/predictions`, `/models`, `/simulation`, and `/methodology`. A React + Plotly dashboard presents market history, classified news, held-out probabilities, model metrics, simulated portfolios, methodology, and limitations. Requests never train models or rerun simulations, so displayed values retain the saved-artifact contract.

## Failed and inconclusive experiments

1. **Rare-event classification:** high raw accuracy concealed weak rare-class recall.
2. **News features:** the +0.008 balanced-accuracy change is too small to distinguish from noise and did not improve ROC-AUC.
3. **Momentum strategy:** frequent trading and costs destroyed the rule.
4. **PPO:** results varied materially by seed and did not reliably beat cash.

These are project results, not defects to hide. They answer the research question more honestly than a selectively reported positive run.

## Limitations

- One instrument and one historical window do not establish generalisation.
- The news corpus is highly imbalanced, headline-only, day-precision, and ends before the market series.
- Geopolitical labels and sentiment describe text, not causal market impact.
- The simulator omits slippage, spread, liquidity, market impact, taxes, dividends, partial fills, shorting, and portfolio construction.
- The held-out period has only 182 sessions; Sharpe estimates and small metric differences are unstable.
- PPO can overfit paths, seeds, reward design, and execution assumptions.
- The dashboard is an artifact viewer, not a live predictor or trading system.

## Reproduction

Run Phases 1–7 in the order documented in `README.md`, then execute `pytest -q` in the project virtual environment. Experiment artifacts are under `reports/experiment_results/`; the dashboard starts with:

```bash
python -m stock_project.backend.app --config configs/config.example.yaml
```

Open `http://127.0.0.1:5000`. React and Plotly are pinned CDN assets, so the browser needs internet access for charts; API reads and automated tests are offline.
