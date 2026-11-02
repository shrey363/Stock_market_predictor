# Dataset Notes

Every dataset used by this project is recorded here **before** it is used: source,
licence and terms, retrieval date, columns, date coverage, known gaps, and the
preprocessing applied. Numbers below were produced by the Phase 1 pipeline and are
reproduced in [`reports/experiment_results/phase1_data_report.json`](../reports/experiment_results/phase1_data_report.json).

Retrieval date for both datasets: **2026-10-13**. Raw files live under `data/raw/`
(git-ignored) and are pinned by SHA-256 in [`data/manifest.json`](../data/manifest.json).

> Simulation and research only. No financial advice, no live trading, and no claim
> that any of this data predicts prices.

---

## 1. Market data — SPY daily OHLCV

| Field | Value |
|---|---|
| Instrument | `SPY` (SPDR S&P 500 ETF Trust) — one broad index proxy, chosen because the first version stays small and correct |
| Source | Yahoo Finance, accessed through the `yfinance` Python package `1.7.0` |
| Access | No API key or account; `yfinance` fetches per call and the result is cached locally |
| Requested window | `2018-01-01` … `2022-12-31`, interval `1d` |
| Rows retrieved | 1,259 sessions |
| Actual coverage | `2018-01-02` … `2022-12-30` |
| Raw file | `data/raw/market/SPY_1d_20180101_20221231.csv` (260 KB, git-ignored) |
| Cleaned file | `data/processed/market_SPY_1d_clean.csv` (1,259 rows) |

### Licence and terms (checked 2026-10-13)

- `yfinance` itself is open-source (Apache-2.0) and describes itself as a library for
  research and education. **The library licence is not a data licence.**
- The underlying Yahoo Finance service documents its API as intended for *personal
  use*, and Yahoo's terms restrict automated collection and commercial reuse without
  prior permission. No explicit academic-research carve-out was found.
- **Conclusion:** this project uses the data for non-commercial academic coursework
  only. Raw rows are never committed or redistributed (`data/raw/` is git-ignored),
  and the pipeline records provenance instead of shipping the data. Anyone reusing
  this repository to publish results should re-check Yahoo's terms, or swap in a
  dataset with explicit redistribution rights (the config keeps the source in one
  place to make that swap a one-line change).

### Columns

`symbol, timestamp, open, high, low, close, adj_close, volume, source`

`timestamp` is UTC-aware. `source` is the configured provenance label, carried on
every row so a later merge cannot lose traceability.

### Known gaps and limitations

- Non-trading days (weekends, market holidays) simply have **no row**. Nothing is
  gap-filled or forward-filled; the cleaned table has 0 missing cells and 0
  duplicates in the retrieved snapshot.
- The timestamp is a **UTC date stamp for the session**, not an intraday instant.
  Intraday timing is out of scope for Phase 1.
- Adjusted close is provider-defined; the fetch requests unadjusted prices plus the
  provider's `Adj Close`, and split/dividend action columns are dropped rather than
  stored twice.
- Providers revise history. The manifest pins the retrieved snapshot by SHA-256, so
  "the data changed" is detectable instead of silent.

### Preprocessing applied

Strict ISO-8601 timestamp parsing (ambiguous formats such as `01/02/2020` are
rejected, never guessed) → UTC normalisation → duplicate detection on
`(symbol, timestamp)` → ascending chronological sort → per-column missing-value
reporting. **No imputation of any kind.**

---

## 2. News data — Hugging Face `heegyu/news-category-dataset`

| Field | Value |
|---|---|
| Dataset | `heegyu/news-category-dataset` — mirror of the Kaggle *News Category Dataset* (HuffPost headlines) |
| URL | <https://huggingface.co/datasets/heegyu/news-category-dataset> |
| Revision | `304a05a55bc6abc0446d8fae0d0771716b6a271a` (pinned in the config) |
| File | `data.json`, JSON Lines (one object per line) |
| Licence | **CC BY 4.0** — attribution required, licence link required, changes must be indicated |
| Access | Public and ungated; no API key or account |
| Retrieval | 2026-10-13 — 87,295,572 bytes, SHA-256 `2cfc51fa4a82a96ecbe251fa383737aa1dea2e0a517f61dece633fcc25a4a065` |
| Upstream coverage | `2012-01-28` … `2022-09-23`, **209,527** records, **42** distinct publisher categories |
| Used subset | source category `POLITICS`, window `2018-01-01` … `2022-12-31` inclusive → **5,930** rows (cap 12,000, so no subsampling was needed) |
| Subset coverage | `2018-01-01` … `2022-09-19` |
| Raw file | `data/raw/news/news-category-dataset/data.json` (84 MB, git-ignored) |
| Interim file | `data/interim/news_politics_20180101_20221231.csv` (5,930 rows, 1.9 MB) |
| Cleaned file | `data/processed/news_news-category-dataset_politics_clean.csv` (5,930 rows, 2.3 MB) |

### Attribution (required by CC BY 4.0)

> News Category Dataset (Rishabh Misra, HuffPost headlines), accessed via the Hugging
> Face mirror `heegyu/news-category-dataset` at revision `304a05a5…`, licensed
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes made for this
> project: filtered to the `POLITICS` category and the 2018–2022 date window,
> de-duplicated, added derived columns (`article_id`, `available_from`, `source`,
> `source_url`), renamed `short_description` → `article_text` and `category` →
> `source_category`, parsed dates to UTC. Headline and description text is unmodified.
> The upstream source is the Kaggle dataset by Rishabh Misra; CC BY 4.0 obligations
> follow this copy of the data.

### Columns

Source columns: `link, headline, category, short_description, authors, date`.

Cleaned table (`data/processed/…_clean.csv`):

| Column | Meaning | Status in Phase 1 |
|---|---|---|
| `article_id` | Derived: SHA-1 of `source_url\|headline\|date`, first 16 hex chars — traceability key, not a source field | populated |
| `published_at` | Publication **date** at midnight UTC (the source has no time of day) | populated |
| `available_from` | `published_at + 1 day` — the conservative instant from which the article may be used for features | populated |
| `headline` | Original headline, trimmed only | populated |
| `article_text` | The source's `short_description` (the dataset has no article body) | populated |
| `source` | Publisher host derived from the URL, `www.` stripped | derived |
| `source_url` | Original article link | populated |
| `source_category` | The **publisher's own** category (`POLITICS`) — not our taxonomy | populated |
| `authors` | Source author field | 134/5,930 missing (2.26%) |
| `event_category`, `sentiment_label`, `sentiment_score`, `affected_sector` | Our labels | **left empty by cleaning** — reported as missing, never filled with placeholders. Phase 2 writes `event_category`, `sentiment_label` and `sentiment_score` into the separate *classified* table (§4); this cleaned table stays untouched. |

### Known gaps and limitations

- **Day-level timestamps.** The source gives a calendar date, not a publication time.
  An article dated `D` cannot be proven to have existed before a cutoff on `D`, so the
  cleaned table carries `available_from = D + 1 day` and Phase 4 must use **that**
  column for feature cutoffs. This is a deliberate, conservative look-ahead guard (see
  `docs/ARCHITECTURE.md` §3).
- **Not a geopolitical corpus.** This is general-interest political news. The
  `POLITICS` filter is a *size* filter; it does not make every headline geopolitical,
  and no *human* labeling has happened yet. Phase 2 defines the event taxonomy and measures
  the match rate: the baseline mapped **221 of 5,930 headlines (3.7%)** to one of the five
  event categories, so the corpus is dominated by non-geopolitical domestic politics. That
  is a valid, documentable result — downstream news features start from a very sparse signal.
- **Publisher taxonomy ≠ ours.** The 42 upstream categories are HuffPost's. We keep them
  as `source_category` and never treat them as our event labels.
- **No body text, no labels.** Classification will be headline-based (plus the short
  description), and there are no sentiment or severity labels in the source.
- **Publisher host changed over time:** `huffingtonpost.com` (3,067 rows) and
  `huffpost.com` (2,863 rows). Any per-publisher analysis must treat these as one outlet.
- **Coverage mismatch with the market series.** Market data runs to `2022-12-30`, news
  to `2022-09-19`; 887 calendar days appear in both tables. Phase 3/4 must not assume
  news exists for the final market sessions.
- **`authors` is incomplete** (134 missing rows) — reported, not imputed.

### Preprocessing applied

1. Download the pinned revision once; store it unmodified and hash it.
2. Stream the JSON Lines file, keeping only rows whose source category matches the
   configured filter inside the configured window. Skips are counted and reported
   (rows read, wrong category, outside window, missing/invalid date, malformed lines).
3. De-duplicate on `article_id`, then on `(headline, published_at)`, reporting both counts.
4. Derive `article_id`, `source_url`, `source`, `source_category`, `article_text`,
   `available_from`; set the four label columns to missing.
5. Sort chronologically by `published_at`; report per-column missing values.

---

## 3. Artifacts produced in Phase 1

| File | Rows / size | Committed? | Produced by |
|---|---|---|---|
| `data/raw/market/SPY_1d_20180101_20221231.csv` | 1,259 / 260 KB | no (git-ignored) | `fetch_data` |
| `data/raw/news/news-category-dataset/data.json` | 209,527 / 84 MB | no (git-ignored) | `fetch_data` |
| `data/interim/news_politics_20180101_20221231.csv` | 5,930 / 1.9 MB | no (git-ignored) | `fetch_data` |
| `data/processed/market_SPY_1d_clean.csv` | 1,259 / 260 KB | no (git-ignored) | `inspect_data` |
| `data/processed/news_news-category-dataset_politics_clean.csv` | 5,930 / 2.3 MB | no (git-ignored) | `inspect_data` |
| `data/manifest.json` | 4 dataset entries | **yes** | `fetch_data`, `inspect_data` |
| `reports/experiment_results/phase1_data_report.json` | 12 KB | **yes** | `inspect_data` |

Derived and raw data are git-ignored because they are fully reproducible from the
documented sources with the two commands below — and because the market data is not
ours to redistribute.

### Reproducing Phase 1

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .

python -m stock_project.data.fetch_data   --config configs/config.example.yaml   # network (≈84 MB download)
python -m stock_project.data.inspect_data --config configs/config.example.yaml   # offline
```

---

## 4. Derived artifacts produced in Phase 2

Phase 2 adds machine labels and human-review sheets; it adds **no new dataset**, so the
source and licence rules above still govern everything here (CC BY 4.0 attribution continues
to apply).

| File | Rows / size | Committed? | Produced by |
|---|---|---|---|
| `data/processed/news_classified_news-category-dataset_politics.csv` | 5,930 / 2.5 MB | no (git-ignored) | `nlp.classify_news` |
| `data/annotations/news_review_sample.csv` | 200 | **yes** | `nlp.annotate` |
| `data/annotations/news_review_focus.csv` | 60 | **yes** | `nlp.annotate` |
| `data/annotations/news_review_sample_labelling_log.jsonl` | 0 rows until labelling starts | **yes** | `nlp.labeller` |
| `reports/experiment_results/phase2_classification_report.json` | 24 KB | **yes** | `nlp.classify_news` |
| `reports/experiment_results/phase2_annotation_sheets.json` | 2 KB | **yes** | `nlp.annotate` |
| `reports/experiment_results/phase2_classifier_evaluation.json` | 3 KB | **yes** | `nlp.evaluate_classifier` |

Machine labels written (taxonomy v1.1, headline scope): `other_or_uncertain` 5,709 (96.3%),
`armed_conflict` 96, `diplomatic_development` 78, `trade_restriction` 24, `sanctions` 17,
`energy_or_supply_disruption` 6. These are **classifier outputs, not ground truth**, and are
never presented as human labels. Sentiment (lexicon baseline): 266 positive, 451 negative,
5,213 neutral, mean −0.031.

Human labels: **none yet.** The review sheets ship with an empty `event_category` column and
no prediction visible, and `phase2_classifier_evaluation.json` states that no metrics were
computed rather than reporting placeholder numbers. Labelling is done with
`python -m stock_project.nlp.labeller --reviewer "<name>"`, which writes only what a reviewer
pressed (skips stay blank), rejects anything outside the taxonomy, and appends every change to
the audit log above. `data/annotations/` is committed so the labelling work cannot be lost;
derived data stays git-ignored because it regenerates from the commands below.

### Reproducing Phase 2

```bash
python -m stock_project.nlp.classify_news       --config configs/config.example.yaml   # offline
python -m stock_project.nlp.annotate            --config configs/config.example.yaml   # offline
# human step: fill event_category in data/annotations/news_review_sample.csv
python -m stock_project.nlp.evaluate_classifier --config configs/config.example.yaml   # offline
```

---

## 5. Phase 1 checks (all passing on the retrieved data)

| Check | Result |
|---|---|
| `market_rows_present` | 1,259 session rows |
| `market_chronological` | strictly ascending, no shuffling |
| `market_sessions_unique` | 0 duplicates |
| `market_no_imputation` | no values imputed or forward-filled |
| `news_rows_present` | 5,930 article rows |
| `news_published_at_parsed` | 0 unparsable dates |
| `news_available_from_after_published_at` | every article available from the day after publication |
| `news_article_ids_unique` | 0 duplicates |
| `news_labels_not_invented` | label columns empty by design |
| `market_news_windows_overlap` | 887 shared calendar days |

**Honest summary:** scraping or downloading happened exactly once, the snapshot is
pinned by hash, and the only "surprising" findings are that the two windows do not
align (news stops ~3 months before the market series) and that timestamps are
day-level — both of which later phases must handle rather than paper over.
