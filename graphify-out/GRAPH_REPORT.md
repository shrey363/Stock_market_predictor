# Graph Report - Stock_market_predictor  (2026-11-21)

## Corpus Check
- 70 files · ~71,426 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 10 file(s) not represented in the graph (top: .csv 7, .jsonl 2, (none) 1)

## Summary
- 1045 nodes · 2709 edges · 53 communities (41 shown, 12 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 157 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6d484aa8`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Project Documentation & Invariants
- Interactive Labeller
- Config Validation
- Event Classifier & Taxonomy
- Market Cleaning & Quality
- labeller.py
- Test Fixtures
- News Cleaning
- Market Data Ingestion
- Review Sheet Building
- Technical Indicators
- Classifier Evaluation
- sentiment.py
- Chronological Splits
- inspect_data.py
- news_features.py
- train_news_aware.py
- test_news_data.py
- classify_news.py
- processed:features_news:SPY:1d
- trading
- Manifest: Classified News
- Manifest: Market Data
- Manifest: Raw News
- Manifest: Processed Features
- Manifest: Processed Market
- Manifest: Processed News
- paths.py
- pandas
- Data Inspection
- News Config Section
- Config Root
- news_data.py
- Market Config Section
- _select_and_fit
- Features Config Section
- NLP Config Section
- utc_now_iso
- Model Config Section
- test_classify_news.py
- Manifest Root
- Prediction Config Section
- Repository Root Package

## God Nodes (most connected - your core abstractions)
1. `load_config()` - 73 edges
2. `Config` - 37 edges
3. `SheetStore` - 34 edges
4. `run_comparison()` - 33 edges
5. `relative_to_root()` - 33 edges
6. `DataSourceError` - 31 edges
7. `run_baseline()` - 30 edges
8. `classify_datasets()` - 30 edges
9. `inspect_datasets()` - 29 edges
10. `clean_news_data()` - 29 edges

## Surprising Connections (you probably didn't know these)
- `test_missing_classified_table_raises()` --uses--> `DataNotFetchedError`  [INFERRED]
  tests/test_annotate.py → src/stock_project/common/errors.py
- `test_missing_cleaned_table_raises_with_the_next_command()` --uses--> `DataNotFetchedError`  [INFERRED]
  tests/test_classify_news.py → src/stock_project/common/errors.py
- `test_missing_classified_table_raises()` --uses--> `DataNotFetchedError`  [INFERRED]
  tests/test_evaluate_classifier.py → src/stock_project/common/errors.py
- `test_missing_sheet_raises_with_the_next_command()` --uses--> `DataNotFetchedError`  [INFERRED]
  tests/test_evaluate_classifier.py → src/stock_project/common/errors.py
- `test_missing_cached_data_raises_with_the_next_command()` --uses--> `DataNotFetchedError`  [INFERRED]
  tests/test_inspect_data.py → src/stock_project/common/errors.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **No-Look-Ahead Invariant Set** — docs_architecture_chronological_invariant, docs_architecture_prediction_cutoff, docs_architecture_rl_information_set, docs_dataset_notes_available_from_guard, readme_no_lookahead_rule [INFERRED 0.85]
- **Phase 2 Human Labelling Workflow** — docs_roadmap_phase2, docs_roadmap_human_labelling_blocker, docs_annotation_guidelines_blind_labelling, readme_interactive_labeller [INFERRED 0.90]
- **Seven-Phase Development Roadmap** — docs_roadmap_phase1, docs_roadmap_phase2, docs_roadmap_phase3, docs_roadmap_phase4, docs_roadmap_phase5, docs_roadmap_phase6, docs_roadmap_phase7 [INFERRED 0.90]

## Communities (53 total, 12 thin omitted)

### Community 0 - "Project Documentation & Invariants"
Cohesion: 0.05
Nodes (40): Current Project State (AGENTS.md), Annotation Known Limitations, Event Category Definitions v1.1, Planned Read-Only Flask API Surface, Data Contracts, High-Level Data Flow, Feature Table Contract, Invariant to Test Map (+32 more)

### Community 1 - "Interactive Labeller"
Cohesion: 0.08
Nodes (34): create_server(), main(), make_handler(), do_GET(), do_POST(), _send(), _send_json(), SheetStore (+26 more)

### Community 2 - "Config Validation"
Cohesion: 0.05
Nodes (62): _check_cutoff_time(), _check_window(), default_config_path(), FeatureSettings, load_config(), MarketSettings, ModelSettings, NewsSettings (+54 more)

### Community 3 - "Event Classifier & Taxonomy"
Cohesion: 0.06
Nodes (32): _class_imbalance(), classification_input(), classification_summary(), classify_frame(), classify_text(), _conditional_nearby(), EventMatch, _exclusions_present() (+24 more)

### Community 4 - "Market Cleaning & Quality"
Cohesion: 0.08
Nodes (25): _empty_string_missing(), _iso(), missing_counts(), missing_mask(), QualityReport, build_checks(), clean_market_data(), _coerce_timestamp() (+17 more)

### Community 6 - "Test Fixtures"
Cohesion: 0.10
Nodes (14): base_config(), config_dumper(), config_path(), config_writer(), deep_merge(), dump_config(), market_provider_frame(), market_raw() (+6 more)

### Community 7 - "News Cleaning"
Cohesion: 0.16
Nodes (20): clean_news_data(), derive_article_id(), publisher_from_url(), _frame(), _row(), test_article_ids_are_derived_unique_and_stable(), test_article_text_comes_from_short_description(), test_blank_and_missing_headlines_count_as_missing() (+12 more)

### Community 8 - "Market Data Ingestion"
Cohesion: 0.13
Nodes (12): DataSourceError, StockProjectError, normalise_download(), yfinance_downloader(), test_cached_file_is_reused_unless_forced(), fake_downloader(), test_load_market_data_missing_file_raises(), test_load_market_data_round_trips_utc_timestamps() (+4 more)

### Community 9 - "Review Sheet Building"
Cohesion: 0.20
Nodes (18): _blank_labels(), build_focus_sheet(), build_review_sheet(), has_labels(), annotate_report(), classified_frame(), prepare_phase2(), test_cli_exits_zero_and_says_labels_are_empty() (+10 more)

### Community 10 - "Technical Indicators"
Cohesion: 0.14
Nodes (11): build_market_features(), intraday_range(), overnight_gap(), _parse_cutoff_offset(), rolling_volatility(), rsi(), _rsi_from_averages(), simple_returns() (+3 more)

### Community 11 - "Classifier Evaluation"
Cohesion: 0.11
Nodes (18): compute_metrics(), error_patterns(), _evaluate_sheet(), load_labelled_sheet(), naive_majority_baseline(), prepare_labelled_project(), test_confusion_matrix_is_laid_out_as_documented(), test_error_patterns_handle_no_errors() (+10 more)

### Community 12 - "sentiment.py"
Cohesion: 0.06
Nodes (20): _find_hits(), get_sentiment_scorer(), score_frame(), score_text(), sentiment_summary(), SentimentResult, test_empty_and_missing_text_is_neutral(), test_mode_selector_exposes_only_the_implemented_scorer() (+12 more)

### Community 13 - "Chronological Splits"
Cohesion: 0.13
Nodes (9): chronological_split(), ChronologicalSplit, slice_summary(), train_validation_test(), test_boundaries_floor_deterministically(), test_invalid_fractions_are_rejected(), test_periods_are_time_ordered_and_disjoint(), test_ranges_are_contiguous_and_cover_every_row_exactly_once() (+1 more)

### Community 14 - "inspect_data.py"
Cohesion: 0.12
Nodes (11): get_dataset_entry(), make_dataset_entry(), read_manifest(), sha256_file(), write_manifest(), _iso_day(), fetch_market_data(), load_market_data() (+3 more)

### Community 15 - "news_features.py"
Cohesion: 0.21
Nodes (5): _as_utc(), build_news_features(), news_feature_columns(), news_feature_definitions(), test_feature_columns_and_definitions_agree()

### Community 16 - "train_news_aware.py"
Cohesion: 0.14
Nodes (9): Config, DataNotFetchedError, market_processed_path(), feature_definitions(), features_processed_path(), majority_class(), run_baseline(), news_features_processed_path() (+1 more)

### Community 17 - "test_news_data.py"
Cohesion: 0.13
Nodes (19): _even_subsample(), extract_news_subset(), iter_jsonl(), load_news_subset(), _stream_download(), write_news_subset(), _copy_fixture_downloader(), test_cap_is_deterministic_and_spreads_coverage() (+11 more)

### Community 18 - "classify_news.py"
Cohesion: 0.14
Nodes (16): configure_cli_logging(), ensure_directories(), relative_to_root(), news_processed_path(), main(), main(), annotate(), main() (+8 more)

### Community 19 - "processed:features_news:SPY:1d"
Cohesion: 0.12
Nodes (17): processed:features_news:SPY:1d, columns, dataset_id, date_coverage, file, key, license, license_url (+9 more)

### Community 20 - "trading"
Cohesion: 0.50
Nodes (4): trading, initial_capital, position_fraction, transaction_cost_bps

### Community 21 - "Manifest: Classified News"
Cohesion: 0.12
Nodes (17): columns, dataset_id, date_coverage, file, key, license, license_url, name (+9 more)

### Community 22 - "Manifest: Market Data"
Cohesion: 0.12
Nodes (17): market:SPY:1d, columns, dataset_id, date_coverage, file, key, license, license_url (+9 more)

### Community 23 - "Manifest: Raw News"
Cohesion: 0.12
Nodes (17): news:heegyu/news-category-dataset, columns, dataset_id, date_coverage, file, key, license, license_url (+9 more)

### Community 24 - "Manifest: Processed Features"
Cohesion: 0.12
Nodes (17): processed:features:SPY:1d, columns, dataset_id, date_coverage, file, key, license, license_url (+9 more)

### Community 25 - "Manifest: Processed Market"
Cohesion: 0.12
Nodes (17): processed:market:SPY:1d, columns, dataset_id, date_coverage, file, key, license, license_url (+9 more)

### Community 26 - "Manifest: Processed News"
Cohesion: 0.12
Nodes (17): processed:news:heegyu/news-category-dataset, columns, dataset_id, date_coverage, file, key, license, license_url (+9 more)

### Community 28 - "paths.py"
Cohesion: 0.19
Nodes (14): all_data_dirs(), annotations_dir(), configs_dir(), data_dir(), figures_dir(), interim_dir(), manifest_path(), processed_dir() (+6 more)

### Community 29 - "pandas"
Cohesion: 0.15
Nodes (13): _market(), test_feature_table_contract_and_warmup_drop(), test_features_do_depend_on_earlier_prices(), test_intraday_range_and_overnight_gap_are_exact(), test_mutating_the_predicted_session_leaves_its_features_untouched(), test_rolling_volatility_is_sample_stdev_of_returns(), test_rsi_edge_cases_are_reported_not_guessed(), test_rsi_matches_hand_computation() (+5 more)

### Community 30 - "Data Inspection"
Cohesion: 0.32
Nodes (9): inspect_datasets(), main(), prepare_cached_project(), test_artifact_paths_are_project_relative(), test_cli_exits_zero_and_prints_the_summary(), test_missing_cached_data_raises_with_the_next_command(), test_processed_tables_and_manifest_are_written(), test_report_is_written_and_every_check_passes() (+1 more)

### Community 31 - "News Config Section"
Cohesion: 0.17
Nodes (12): news, dataset_id, dataset_url, end, license, license_url, max_articles, revision (+4 more)

### Community 32 - "Config Root"
Cohesion: 0.25
Nodes (7): data, project, name, random_seed, splits, train_fraction, validation_fraction

### Community 33 - "news_data.py"
Cohesion: 0.17
Nodes (10): fetch_all(), fetch_market(), fetch_news(), main(), dataset_slug(), fetch_news_dataset(), raw_news_path(), resolve_url() (+2 more)

### Community 34 - "Market Config Section"
Cohesion: 0.29
Nodes (7): market, end, interval, source, start, symbol, timezone

### Community 35 - "_select_and_fit"
Cohesion: 0.27
Nodes (3): fit_logistic(), probabilities(), _select_and_fit()

### Community 36 - "Features Config Section"
Cohesion: 0.29
Nodes (7): features, moving_average_windows, news_lookback_days, return_windows, rsi_window, volatility_window, volume_window

### Community 37 - "NLP Config Section"
Cohesion: 0.33
Nodes (6): nlp, review_sample_seed, review_sample_size, sentiment_mode, taxonomy_version, text_scope

### Community 38 - "utc_now_iso"
Cohesion: 0.36
Nodes (3): is_utc(), utc_now(), utc_now_iso()

### Community 39 - "Model Config Section"
Cohesion: 0.40
Nodes (5): model, c_values, max_iter, selection_metric, type

### Community 40 - "test_classify_news.py"
Cohesion: 0.21
Nodes (12): build_classified_table(), classification_examples(), _examples(), prepare_cleaned_news(), test_classified_table_matches_the_documented_contract(), test_cli_exits_zero_and_prints_the_distribution(), test_empty_headlines_are_counted_and_fall_back(), test_end_to_end_classification_writes_table_and_report() (+4 more)

### Community 41 - "Manifest Root"
Cohesion: 0.50
Nodes (3): datasets, manifest_version, updated_at_utc

### Community 42 - "Prediction Config Section"
Cohesion: 0.67
Nodes (3): prediction, cutoff_time, target

## Knowledge Gaps
- **3 isolated node(s):** `stock-project`, `Seven Development Milestones`, `Limitations and Responsible Use`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 407 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_config()` connect `Config Validation` to `news_data.py`, `Interactive Labeller`, `labeller.py`, `test_classify_news.py`, `Review Sheet Building`, `Classifier Evaluation`, `inspect_data.py`, `train_news_aware.py`, `classify_news.py`, `Data Inspection`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Are the 22 inferred relationships involving `Config` (e.g. with `fetch_all()` and `fetch_market()`) actually correct?**
  _`Config` has 22 INFERRED edges - model-reasoned connections that need verification._
- **What connects `stock-project`, `Seven Development Milestones`, `Limitations and Responsible Use` to the rest of the system?**
  _3 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Project Documentation & Invariants` be split into smaller, more focused modules?**
  _Cohesion score 0.05213089802130898 - nodes in this community are weakly interconnected._
- **Why does `SheetStore` connect `Interactive Labeller` to `train_news_aware.py`, `labeller.py`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._
- **Should `Interactive Labeller` be split into smaller, more focused modules?**
  _Cohesion score 0.0798611111111111 - nodes in this community are weakly interconnected._
- **Why does `relative_to_root()` connect `classify_news.py` to `news_data.py`, `Config Validation`, `Interactive Labeller`, `labeller.py`, `Classifier Evaluation`, `inspect_data.py`, `train_news_aware.py`, `paths.py`, `Data Inspection`?**
  _High betweenness centrality (0.016) - this node is a cross-community bridge._