# Graph Report - Stock_market_predictor  (2026-11-21)

## Corpus Check
- 66 files · ~63,168 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 8 file(s) not represented in the graph (top: .csv 5, .jsonl 2, (none) 1)

## Summary
- 970 nodes · 2461 edges · 51 communities (42 shown, 9 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 152 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Project Documentation & Invariants
- Interactive Labeller
- Config Validation
- Event Classifier & Taxonomy
- Market Cleaning & Quality
- Analysis & Reporting CLIs
- Test Fixtures
- News Cleaning
- Market Data Ingestion
- Review Sheet Building
- Technical Indicators
- Classifier Evaluation
- Lexicon Sentiment
- Chronological Splits
- Annotation & Evaluation Flow
- Data Paths
- Baseline Training
- News Ingestion
- News Classification Pipeline
- News Dataset Loading
- Taxonomy Tests
- Manifest: Classified News
- Manifest: Market Data
- Manifest: Raw News
- Manifest: Processed Features
- Manifest: Processed Market
- Manifest: Processed News
- Manifest & Provenance
- Frame Scoring Helpers
- Indicator Tests
- Data Inspection
- News Config Section
- Config Root
- Data Fetch CLI
- Market Config Section
- Return & SMA Features
- Features Config Section
- NLP Config Section
- Logistic Fit Helpers
- Model Config Section
- Classification Examples
- Manifest Root
- Prediction Config Section
- Splits Config Section
- Repository Root Package

## God Nodes (most connected - your core abstractions)
1. `load_config()` - 66 edges
2. `SheetStore` - 34 edges
3. `Config` - 32 edges
4. `relative_to_root()` - 30 edges
5. `classify_datasets()` - 30 edges
6. `inspect_datasets()` - 29 edges
7. `run_baseline()` - 29 edges
8. `clean_news_data()` - 29 edges
9. `DataSourceError` - 28 edges
10. `DataNotFetchedError` - 25 edges

## Surprising Connections (you probably didn't know these)
- `annotate_report()` --uses--> `Config`  [INFERRED]
  tests/test_annotate.py → src/stock_project/common/config.py
- `classified_frame()` --uses--> `Config`  [INFERRED]
  tests/test_annotate.py → src/stock_project/common/config.py
- `prepare_phase2()` --uses--> `Config`  [INFERRED]
  tests/test_annotate.py → src/stock_project/common/config.py
- `prepare_cleaned_news()` --uses--> `Config`  [INFERRED]
  tests/test_classify_news.py → src/stock_project/common/config.py
- `prepare_labelled_project()` --uses--> `Config`  [INFERRED]
  tests/test_evaluate_classifier.py → src/stock_project/common/config.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Seven-Phase Development Roadmap** — docs_roadmap_phase1, docs_roadmap_phase2, docs_roadmap_phase3, docs_roadmap_phase4, docs_roadmap_phase5, docs_roadmap_phase6, docs_roadmap_phase7 [INFERRED 0.90]
- **No-Look-Ahead Invariant Set** — docs_architecture_chronological_invariant, docs_architecture_prediction_cutoff, docs_architecture_rl_information_set, docs_dataset_notes_available_from_guard, readme_no_lookahead_rule [INFERRED 0.85]
- **Phase 2 Human Labelling Workflow** — docs_roadmap_phase2, docs_roadmap_human_labelling_blocker, docs_annotation_guidelines_blind_labelling, readme_interactive_labeller [INFERRED 0.90]

## Communities (51 total, 9 thin omitted)

### Community 0 - "Project Documentation & Invariants"
Cohesion: 0.05
Nodes (40): Current Project State (AGENTS.md), Annotation Known Limitations, Event Category Definitions v1.1, Planned Read-Only Flask API Surface, Data Contracts, High-Level Data Flow, Feature Table Contract, Invariant to Test Map (+32 more)

### Community 1 - "Interactive Labeller"
Cohesion: 0.08
Nodes (34): create_server(), main(), make_handler(), do_GET(), do_POST(), _send(), _send_json(), SheetStore (+26 more)

### Community 2 - "Config Validation"
Cohesion: 0.08
Nodes (41): _check_cutoff_time(), _check_window(), default_config_path(), FeatureSettings, load_config(), MarketSettings, ModelSettings, NewsSettings (+33 more)

### Community 3 - "Event Classifier & Taxonomy"
Cohesion: 0.06
Nodes (32): _class_imbalance(), classification_input(), classification_summary(), classify_frame(), classify_text(), _conditional_nearby(), EventMatch, _exclusions_present() (+24 more)

### Community 4 - "Market Cleaning & Quality"
Cohesion: 0.07
Nodes (25): _empty_string_missing(), _iso(), missing_counts(), missing_mask(), QualityReport, build_checks(), clean_market_data(), _coerce_timestamp() (+17 more)

### Community 5 - "Analysis & Reporting CLIs"
Cohesion: 0.13
Nodes (14): configure_cli_logging(), get_logger(), relative_to_root(), results_dir(), utc_now(), utc_now_iso(), _iso_day(), main() (+6 more)

### Community 6 - "Test Fixtures"
Cohesion: 0.10
Nodes (14): base_config(), config_dumper(), config_path(), config_writer(), deep_merge(), dump_config(), market_provider_frame(), market_raw() (+6 more)

### Community 7 - "News Cleaning"
Cohesion: 0.13
Nodes (22): DataSourceError, StockProjectError, clean_news_data(), derive_article_id(), publisher_from_url(), _frame(), _row(), test_article_ids_are_derived_unique_and_stable() (+14 more)

### Community 8 - "Market Data Ingestion"
Cohesion: 0.12
Nodes (16): is_utc(), fetch_market_data(), load_market_data(), market_file_path(), market_filename(), normalise_download(), yfinance_downloader(), test_cached_file_is_reused_unless_forced() (+8 more)

### Community 9 - "Review Sheet Building"
Cohesion: 0.22
Nodes (17): _blank_labels(), build_focus_sheet(), build_review_sheet(), annotate_report(), classified_frame(), prepare_phase2(), test_cli_exits_zero_and_says_labels_are_empty(), test_existing_labels_are_never_overwritten() (+9 more)

### Community 10 - "Technical Indicators"
Cohesion: 0.12
Nodes (14): build_market_features(), feature_definitions(), intraday_range(), overnight_gap(), _parse_cutoff_offset(), rolling_volatility(), rsi(), _rsi_from_averages() (+6 more)

### Community 11 - "Classifier Evaluation"
Cohesion: 0.13
Nodes (15): compute_metrics(), error_patterns(), _evaluate_sheet(), load_labelled_sheet(), naive_majority_baseline(), test_confusion_matrix_is_laid_out_as_documented(), test_error_patterns_handle_no_errors(), test_error_patterns_summarise_confusions() (+7 more)

### Community 12 - "Lexicon Sentiment"
Cohesion: 0.13
Nodes (10): _find_hits(), get_sentiment_scorer(), score_text(), SentimentResult, test_empty_and_missing_text_is_neutral(), test_mode_selector_exposes_only_the_implemented_scorer(), test_negation_flips_the_following_term(), test_score_is_always_within_bounds() (+2 more)

### Community 13 - "Chronological Splits"
Cohesion: 0.13
Nodes (9): chronological_split(), ChronologicalSplit, slice_summary(), train_validation_test(), test_boundaries_floor_deterministically(), test_invalid_fractions_are_rejected(), test_periods_are_time_ordered_and_disjoint(), test_ranges_are_contiguous_and_cover_every_row_exactly_once() (+1 more)

### Community 14 - "Annotation & Evaluation Flow"
Cohesion: 0.19
Nodes (11): DataNotFetchedError, annotate(), has_labels(), review_sheet_path(), classified_path(), evaluate_classifier(), prepare_labelled_project(), test_end_to_end_evaluation_writes_metrics_and_error_list() (+3 more)

### Community 15 - "Data Paths"
Cohesion: 0.23
Nodes (12): all_data_dirs(), annotations_dir(), configs_dir(), data_dir(), figures_dir(), interim_dir(), processed_dir(), project_root() (+4 more)

### Community 16 - "Baseline Training"
Cohesion: 0.15
Nodes (6): Config, ensure_directories(), market_processed_path(), features_processed_path(), _majority_class(), run_baseline()

### Community 17 - "News Ingestion"
Cohesion: 0.18
Nodes (14): _even_subsample(), extract_news_subset(), load_news_subset(), _copy_fixture_downloader(), test_cap_is_deterministic_and_spreads_coverage(), test_extract_subset_category_filter_is_case_insensitive(), test_extract_subset_counts_are_exact(), test_extract_subset_reports_what_it_skipped() (+6 more)

### Community 18 - "News Classification Pipeline"
Cohesion: 0.29
Nodes (12): news_processed_path(), build_classified_table(), classify_datasets(), prepare_cleaned_news(), test_classified_table_matches_the_documented_contract(), test_cli_exits_zero_and_prints_the_distribution(), test_empty_headlines_are_counted_and_fall_back(), test_end_to_end_classification_writes_table_and_report() (+4 more)

### Community 19 - "News Dataset Loading"
Cohesion: 0.15
Nodes (9): fetch_news_dataset(), iter_jsonl(), raw_news_path(), resolve_url(), _stream_download(), subset_filename(), subset_path(), test_iter_jsonl_skips_blank_lines_and_reports_malformed_ones() (+1 more)

### Community 20 - "Taxonomy Tests"
Cohesion: 0.12
Nodes (4): _spec_categories(), test_categories_match_the_specification_exactly(), test_the_uncertain_category_has_no_keywords(), test_uncertain_category_is_last_in_priority_order()

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

### Community 27 - "Manifest & Provenance"
Cohesion: 0.23
Nodes (6): get_dataset_entry(), make_dataset_entry(), read_manifest(), sha256_file(), write_manifest(), manifest_path()

### Community 28 - "Frame Scoring Helpers"
Cohesion: 0.20
Nodes (9): classification_metrics(), score_frame(), sentiment_summary(), test_score_frame_adds_the_documented_columns_and_keeps_rows(), test_score_frame_rejects_a_missing_column(), test_summary_reports_labels_and_framing_note(), test_summary_requires_sentiment_columns(), test_auc_is_computed_when_scores_are_supplied() (+1 more)

### Community 29 - "Indicator Tests"
Cohesion: 0.19
Nodes (6): _market(), test_features_do_depend_on_earlier_prices(), test_mutating_the_predicted_session_leaves_its_features_untouched(), test_rolling_volatility_is_sample_stdev_of_returns(), test_rsi_matches_hand_computation(), test_series_shorter_than_the_warmup_is_refused()

### Community 30 - "Data Inspection"
Cohesion: 0.35
Nodes (8): inspect_datasets(), prepare_cached_project(), test_artifact_paths_are_project_relative(), test_cli_exits_zero_and_prints_the_summary(), test_missing_cached_data_raises_with_the_next_command(), test_processed_tables_and_manifest_are_written(), test_report_is_written_and_every_check_passes(), test_report_numbers_match_the_fixtures()

### Community 31 - "News Config Section"
Cohesion: 0.17
Nodes (12): news, dataset_id, dataset_url, end, license, license_url, max_articles, revision (+4 more)

### Community 32 - "Config Root"
Cohesion: 0.22
Nodes (8): data, project, name, random_seed, trading, initial_capital, position_fraction, transaction_cost_bps

### Community 33 - "Data Fetch CLI"
Cohesion: 0.39
Nodes (5): fetch_all(), fetch_market(), fetch_news(), main(), write_news_subset()

### Community 34 - "Market Config Section"
Cohesion: 0.29
Nodes (7): market, end, interval, source, start, symbol, timezone

### Community 35 - "Return & SMA Features"
Cohesion: 0.29
Nodes (5): simple_returns(), sma_ratio(), test_invalid_windows_are_rejected(), test_simple_returns_are_exact(), test_sma_ratio_is_distance_from_the_moving_average()

### Community 36 - "Features Config Section"
Cohesion: 0.33
Nodes (6): features, moving_average_windows, return_windows, rsi_window, volatility_window, volume_window

### Community 37 - "NLP Config Section"
Cohesion: 0.33
Nodes (6): nlp, review_sample_seed, review_sample_size, sentiment_mode, taxonomy_version, text_scope

### Community 39 - "Model Config Section"
Cohesion: 0.40
Nodes (5): model, c_values, max_iter, selection_metric, type

### Community 41 - "Manifest Root"
Cohesion: 0.50
Nodes (3): datasets, manifest_version, updated_at_utc

### Community 42 - "Prediction Config Section"
Cohesion: 0.67
Nodes (3): prediction, cutoff_time, target

### Community 43 - "Splits Config Section"
Cohesion: 0.67
Nodes (3): splits, train_fraction, validation_fraction

## Knowledge Gaps
- **3 isolated node(s):** `stock-project`, `Seven Development Milestones`, `Limitations and Responsible Use`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 370 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_config()` connect `Config Validation` to `Data Fetch CLI`, `Interactive Labeller`, `Analysis & Reporting CLIs`, `Test Fixtures`, `Review Sheet Building`, `Annotation & Evaluation Flow`, `Baseline Training`, `News Classification Pipeline`, `Data Inspection`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Are the 17 inferred relationships involving `SheetStore` (e.g. with `DataNotFetchedError` and `server()`) actually correct?**
  _`SheetStore` has 17 INFERRED edges - model-reasoned connections that need verification._
- **What connects `stock-project`, `Seven Development Milestones`, `Limitations and Responsible Use` to the rest of the system?**
  _3 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Project Documentation & Invariants` be split into smaller, more focused modules?**
  _Cohesion score 0.05213089802130898 - nodes in this community are weakly interconnected._
- **Why does `SheetStore` connect `Interactive Labeller` to `Analysis & Reporting CLIs`, `Annotation & Evaluation Flow`?**
  _High betweenness centrality (0.019) - this node is a cross-community bridge._
- **Should `Interactive Labeller` be split into smaller, more focused modules?**
  _Cohesion score 0.0798611111111111 - nodes in this community are weakly interconnected._
- **Why does `DataNotFetchedError` connect `Annotation & Evaluation Flow` to `Interactive Labeller`, `Config Validation`, `Analysis & Reporting CLIs`, `News Cleaning`, `Review Sheet Building`, `Classifier Evaluation`, `Baseline Training`, `News Classification Pipeline`, `Data Inspection`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._