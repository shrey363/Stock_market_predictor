"""CLI: train the Phase 4 market+news model and compare it with the Phase 3 baseline.

    python -m stock_project.models.train_news_aware --config configs/config.example.yaml

Reads the cleaned market table (Phase 1) and the classified news table (Phase 2),
builds the Phase 3 market features plus the Phase 4 news features, uses the **same**
chronological split as Phase 3, and trains two models with the same family and
settings: market-only and market-plus-news. Both select their regularisation
strength on validation only, refit on train + validation, and are evaluated once
on the identical held-out test period.

This is the project's core research question. A "news features add nothing" result
is a valid outcome and is reported as such.

Artifacts (invariant 7):

* ``data/processed/features_<SYMBOL>_<interval>_news.csv`` — the combined feature
  store (market + news features + target), also recorded in ``data/manifest.json``.
* ``reports/experiment_results/phase4_news_comparison_report.json`` — config,
  feature definitions, split summaries, both models' metrics, coefficients,
  checks and limitations.
* ``reports/experiment_results/phase4_test_predictions.csv`` — one row per held-out
  session with both models' probabilities, so every number traces to a date.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..common.config import Config, load_config
from ..common.errors import DataNotFetchedError
from ..common.logging_utils import configure_cli_logging, get_logger
from ..common.manifest import make_dataset_entry, write_manifest
from ..common.paths import ensure_directories, processed_dir, relative_to_root, results_dir
from ..common.timeutils import utc_now_iso
from ..data.inspect_data import market_processed_path
from ..features.chronological_splits import chronological_split, train_validation_test
from ..features.news_features import (
    NO_NEWS_COLUMN,
    build_news_features,
    news_feature_columns,
    news_feature_definitions,
)
from ..features.technical_indicators import (
    CONTEXT_COLUMNS,
    TARGET_COLUMNS,
    build_market_features,
    feature_definitions,
)
from ..nlp.classify_news import classified_path
from .train_baseline import (
    classification_metrics,
    fit_logistic,
    majority_class,
    probabilities,
)

logger = get_logger("stock_project.models.train_news_aware")

REPORT_FILENAME = "phase4_news_comparison_report.json"
PREDICTIONS_FILENAME = "phase4_test_predictions.csv"

#: Metric used to compare the models, in the report.
COMPARISON_METRIC = "balanced_accuracy"


def news_features_processed_path(config: Config, *, root: Path | None = None) -> Path:
    return processed_dir(root) / f"features_{config.market.symbol.upper()}_{config.market.interval}_news.csv"


def _select_and_fit(
    features: pd.DataFrame,
    feature_columns: list[str],
    split,
    config: Config,
) -> dict[str, Any]:
    """Run one model family over one feature set on the shared chronological split."""
    train_rows = features.iloc[split.train.start : split.train.stop]
    validation_rows = features.iloc[split.validation.start : split.validation.stop]
    test_rows = features.iloc[split.test.start : split.test.stop]

    train_features, train_target = train_rows[feature_columns], train_rows["target"]
    validation_features, validation_target = validation_rows[feature_columns], validation_rows["target"]
    test_features, test_target = test_rows[feature_columns], test_rows["target"]

    selection: list[dict[str, Any]] = []
    best_c = None
    best_score = None
    for c in config.model.c_values:
        candidate = fit_logistic(
            train_features, train_target, c=c, max_iter=config.model.max_iter, seed=config.project.random_seed
        )
        candidate_metrics = classification_metrics(
            validation_target,
            candidate.predict(validation_features),
            probabilities(candidate, validation_features),
        )
        score = candidate_metrics[config.model.selection_metric]
        selection.append({"c": float(c), "validation": candidate_metrics})
        if best_score is None or score > best_score + 1e-12:
            best_score, best_c = score, float(c)
    if best_c is None:  # pragma: no cover - config validation prevents an empty grid
        raise AssertionError("no regularisation strength was evaluated")

    fit_features = pd.concat([train_features, validation_features])
    fit_target = pd.concat([train_target, validation_target])
    final_model = fit_logistic(
        fit_features, fit_target, c=best_c, max_iter=config.model.max_iter, seed=config.project.random_seed
    )

    test_probability = probabilities(final_model, test_features)
    test_prediction = pd.Series(final_model.predict(test_features), index=test_features.index)
    coefficients = {
        column: round(float(value), 6)
        for column, value in zip(feature_columns, final_model.named_steps["model"].coef_[0])
    }
    return {
        "feature_columns": list(feature_columns),
        "selection_metric": config.model.selection_metric,
        "selection_on_validation": selection,
        "selected_c": best_c,
        "fit_rows": int(len(fit_features)),
        "validation_of_selected_c": next(entry["validation"] for entry in selection if entry["c"] == best_c),
        "test": classification_metrics(test_target, test_prediction, test_probability),
        "standardised_coefficients": dict(sorted(coefficients.items(), key=lambda item: abs(item[1]), reverse=True)),
        "_test_probability": test_probability,
        "_test_prediction": test_prediction,
        "_test_target": test_target,
    }


def run_comparison(config: Config, *, root: Path | None = None, write: bool = True) -> dict[str, Any]:
    """Build both feature sets, train both models, return the comparison report."""
    market_file = market_processed_path(config, root=root)
    if not market_file.exists():
        raise DataNotFetchedError(
            f"no cleaned market table at {market_file}; run: "
            "python -m stock_project.data.inspect_data --config configs/config.example.yaml"
        )
    news_file = classified_path(config, root=root)
    if not news_file.exists():
        raise DataNotFetchedError(
            f"no classified news table at {news_file}; run: "
            "python -m stock_project.nlp.classify_news --config configs/config.example.yaml"
        )

    market = pd.read_csv(market_file, dtype={"symbol": "string"})
    news = pd.read_csv(news_file)

    market_definitions = feature_definitions(
        return_windows=config.features.return_windows,
        moving_average_windows=config.features.moving_average_windows,
        rsi_window=config.features.rsi_window,
        volatility_window=config.features.volatility_window,
        volume_window=config.features.volume_window,
    )
    market_features, market_quality = build_market_features(
        market,
        return_windows=config.features.return_windows,
        moving_average_windows=config.features.moving_average_windows,
        rsi_window=config.features.rsi_window,
        volatility_window=config.features.volatility_window,
        volume_window=config.features.volume_window,
        cutoff_time=config.prediction.cutoff_time,
    )
    news_features, news_quality = build_news_features(
        market_features, news, lookback_days=config.features.news_lookback_days
    )

    market_columns = list(market_definitions)
    news_columns = news_feature_columns()
    news_definitions = news_feature_definitions(config.features.news_lookback_days)
    if set(market_columns) & set(news_columns):
        raise AssertionError(f"market and news feature names overlap: {sorted(set(market_columns) & set(news_columns))}")

    features = pd.concat([market_features.reset_index(drop=True), news_features.reset_index(drop=True)], axis=1)
    absent = [column for column in market_columns + news_columns if column not in features.columns]
    if absent:  # pragma: no cover - defensive contract check
        raise AssertionError(f"feature definitions name columns the table lacks: {absent}")

    split = chronological_split(
        len(features),
        train_fraction=config.splits.train_fraction,
        validation_fraction=config.splits.validation_fraction,
    )
    split_summaries = train_validation_test(features, split)
    split_summaries["test"]["share_of_rows"] = round(len(split.test) / len(features), 6)

    train_target = features.iloc[split.train.start : split.train.stop]["target"]
    majority = majority_class(train_target)
    test_target = features.iloc[split.test.start : split.test.stop]["target"]
    naive_test = classification_metrics(test_target, pd.Series([majority] * len(test_target)))

    market_only = _select_and_fit(features, market_columns, split, config)
    market_news = _select_and_fit(features, market_columns + news_columns, split, config)

    no_news_sessions = int(features[NO_NEWS_COLUMN].sum())
    available_from = pd.to_datetime(news["available_from"], utc=True)
    news_start = str(available_from.min().date()) if len(news) else None
    news_end = str(available_from.max().date()) if len(news) else None

    checks = [
        {
            "name": "market_rows_loaded",
            "passed": len(market) > 0,
            "detail": f"{len(market)} cleaned session(s) read from {relative_to_root(market_file, root)}",
        },
        {
            "name": "news_articles_loaded",
            "passed": len(news) > 0,
            "detail": f"{len(news)} classified article(s) read from {relative_to_root(news_file, root)}",
        },
        {
            "name": "news_features_complete",
            "passed": not bool(features[news_columns].isna().any().any())
            and set(features[NO_NEWS_COLUMN].unique()) <= {0, 1},
            "detail": f"every session row has all {len(news_columns)} news features; no_news is 0 or 1, never missing",
        },
        {
            "name": "features_chronological",
            "passed": bool(features["session_date"].is_monotonic_increasing),
            "detail": "feature rows are in ascending session order; nothing was shuffled",
        },
        {
            "name": "features_no_missing_values",
            "passed": not bool(features[market_columns + news_columns].isna().any().any()),
            "detail": "every kept row has a value for every market and news feature; nothing was imputed",
        },
        {
            "name": "split_periods_ordered_and_disjoint",
            "passed": (
                split.train.stop == split.validation.start
                and split.validation.stop == split.test.start
                and split.train.start == 0
                and split.test.stop == len(features)
            ),
            "detail": (
                f"train={len(split.train)} validation={len(split.validation)} test={len(split.test)} contiguous rows "
                "in time order, covering every row exactly once"
            ),
        },
        {
            "name": "target_matches_return_sign",
            "passed": bool(((features["next_session_return"] > 0).astype("int64") == features["target"]).all()),
            "detail": "target is 1 exactly when the predicted session's return is above zero",
        },
        {
            "name": "features_exclude_target",
            "passed": not set(market_columns + news_columns) & (set(CONTEXT_COLUMNS) | set(TARGET_COLUMNS)),
            "detail": "no context or target column is used as a model input",
        },
        {
            "name": "both_models_fit_without_test_rows",
            "passed": (
                market_only["fit_rows"] == len(split.train) + len(split.validation)
                and market_news["fit_rows"] == len(split.train) + len(split.validation)
            ),
            "detail": (
                f"both models were fit on {market_only['fit_rows']} train+validation rows; "
                f"the {len(split.test)} test rows were not used for fitting or selection"
            ),
        },
        {
            "name": "same_test_rows_for_both_models",
            "passed": bool(
                (market_only["_test_probability"].index == market_news["_test_probability"].index).all()
                and (market_only["_test_target"] == market_news["_test_target"]).all()
            ),
            "detail": "market-only and market+news are scored on the same held-out sessions, in the same order",
        },
    ]

    comparison = {
        "metric": COMPARISON_METRIC,
        "naive_majority": naive_test[COMPARISON_METRIC],
        "market_only": market_only["test"][COMPARISON_METRIC],
        "market_plus_news": market_news["test"][COMPARISON_METRIC],
        "news_minus_market_only": round(
            float(market_news["test"][COMPARISON_METRIC]) - float(market_only["test"][COMPARISON_METRIC]), 6
        ),
        "news_beats_market_only_on_test": bool(
            market_news["test"][COMPARISON_METRIC] > market_only["test"][COMPARISON_METRIC]
        ),
        "note": (
            "A single held-out period of a few hundred sessions is a small sample: a difference of a few points is "
            "indistinguishable from noise. This is a fair comparison of two baselines, not evidence of predictive "
            "skill or profitability."
        ),
    }

    report: dict[str, Any] = {
        "generated_at_utc": utc_now_iso(),
        "phase": 4,
        "phase_name": "Add geopolitical features",
        "config": config.to_dict(),
        "inputs": {
            "market_table": relative_to_root(market_file, root),
            "market_rows": int(len(market)),
            "classified_news_table": relative_to_root(news_file, root),
            "news_articles": int(len(news)),
            "news_available_from_range": (news_start, news_end),
            "features_table": relative_to_root(news_features_processed_path(config, root=root), root),
        },
        "cutoff": {
            "decision_rule": (
                "A row for session t is decided at that session's timestamp and every market feature uses bars up to "
                "and including t-1 only; values are computed at the close of their bar and shifted forward one row."
            ),
            "news_rule": (
                f"news_cutoff_utc = session date at {config.prediction.cutoff_time} UTC. Only articles whose "
                f"available_from falls in (news_cutoff_utc - {config.features.news_lookback_days} day(s), "
                "news_cutoff_utc] are aggregated; day-precision news published the day before at the latest is "
                "therefore usable."
            ),
            "target": (
                "target = 1 when close[t] > close[t-1] for the predicted session t; next_session_return records "
                "the exact return it was derived from."
            ),
        },
        "features": {
            "market_columns": market_columns,
            "news_columns": news_columns,
            "market_definitions": market_definitions,
            "news_definitions": news_definitions,
            "news_lookback_days": config.features.news_lookback_days,
            "warmup_rows_dropped": int(market_quality.rows_dropped_missing_required),
            "sessions_without_news": no_news_sessions,
            "sessions_without_news_share": round(no_news_sessions / len(features), 6),
            "notes": list(market_quality.notes) + list(news_quality.notes),
        },
        "target_balance": {
            "rows": int(len(features)),
            "positive": int((features["target"] == 1).sum()),
            "negative": int((features["target"] == 0).sum()),
            "positive_rate": round(float(features["target"].mean()), 6),
        },
        "splits": {
            "fractions": {
                "train": config.splits.train_fraction,
                "validation": config.splits.validation_fraction,
                "test": round(1.0 - config.splits.train_fraction - config.splits.validation_fraction, 6),
            },
            "ranges": split.as_dict(),
            "periods": split_summaries,
        },
        "models": {
            "naive_majority": {
                "predicted_class": majority,
                "definition": "always predict the majority class of the training period",
                "test": naive_test,
            },
            "market_only": {key: value for key, value in market_only.items() if not key.startswith("_")},
            "market_plus_news": {key: value for key, value in market_news.items() if not key.startswith("_")},
        },
        "comparison": comparison,
        "checks": checks,
        "limitations": [
            "One instrument (SPY) and one historical window: results do not generalise to other markets or periods.",
            "The news source ends before the market series: sessions after its last available article carry no_news=1.",
            "News labels come from the Phase 2 keyword baseline, whose agreement with the 200 human-labelled "
            "review rows is weak on rare categories (see phase2_classifier_evaluation.json): classification "
            "error attenuates any signal.",
            "96.3% of the corpus falls to other_or_uncertain and headline scope misses body-only events, so the news signal is sparse.",
            "Sentiment is text tone, not severity or expected market impact.",
            "A single held-out period of a few hundred sessions cannot distinguish a small effect from noise.",
            "This is a research baseline, not financial advice, and it makes no claim of reliable prediction.",
        ],
        "next_step": (
            "Phase 5 — implement the deterministic trading simulator (initial capital, position limits, execution "
            "timing, transaction costs) and its buy-and-hold and rule-based benchmarks."
        ),
    }

    if write:
        ensure_directories(root)
        feature_out = news_features_processed_path(config, root=root)
        features.to_csv(feature_out, index=False, lineterminator="\n")
        write_manifest(
            root,
            key=f"processed:features_news:{config.market.symbol.upper()}:{config.market.interval}",
            entry=make_dataset_entry(
                key=f"processed:features_news:{config.market.symbol.upper()}:{config.market.interval}",
                name=(
                    f"market + news features and next-session target "
                    f"({config.market.symbol.upper()} {config.market.interval})"
                ),
                file_path=feature_out,
                root=root,
                source=(
                    f"{relative_to_root(market_file, root)} + {relative_to_root(news_file, root)}"
                ),
                rows=int(len(features)),
                columns=list(features.columns),
                date_coverage=(str(features["session_date"].iloc[0]), str(features["session_date"].iloc[-1])),
                notes=[
                    "Derived by features.technical_indicators plus features.news_features.",
                    f"News window: (news_cutoff - {config.features.news_lookback_days} day(s), news_cutoff], "
                    "using available_from only (invariant 2).",
                    "Sessions without news carry zero counts and no_news=1; nothing is imputed.",
                ],
            ),
        )

        test_rows = features.iloc[split.test.start : split.test.stop]
        predictions = pd.DataFrame(
            {
                "session_date": test_rows["session_date"].to_numpy(),
                "decision_time_utc": test_rows["decision_time_utc"].to_numpy(),
                "next_session_return": test_rows["next_session_return"].to_numpy(),
                "target": market_only["_test_target"].to_numpy(),
                "probability_up_market_only": market_only["_test_probability"].to_numpy(),
                "prediction_market_only": market_only["_test_prediction"].to_numpy(),
                "probability_up_market_news": market_news["_test_probability"].to_numpy(),
                "prediction_market_news": market_news["_test_prediction"].to_numpy(),
            }
        )
        predictions["correct_market_only"] = (
            predictions["prediction_market_only"] == predictions["target"]
        ).astype(int)
        predictions["correct_market_news"] = (
            predictions["prediction_market_news"] == predictions["target"]
        ).astype(int)
        predictions_path = results_dir(root) / PREDICTIONS_FILENAME
        predictions.to_csv(predictions_path, index=False, lineterminator="\n")

        report_path = results_dir(root) / REPORT_FILENAME
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logger.info("wrote %s", relative_to_root(report_path, root))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the Phase 4 market-only and market+news models and compare them."
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config (default: configs/config.yaml or the example)")
    parser.add_argument("--print-json", action="store_true", help="also print the full report to stdout")
    parser.add_argument("--root", type=Path, default=None, help="project root to read/write (default: this repository)")
    args = parser.parse_args(argv)

    configure_cli_logging()
    config = load_config(args.config)
    root = Path(args.root) if args.root else None
    report = run_comparison(config, root=root)

    periods = report["splits"]["periods"]
    features = report["features"]
    print(
        f"features : {report['target_balance']['rows']} rows "
        f"{periods['train']['first_session']}..{periods['test']['last_session']} "
        f"({len(features['market_columns'])} market + {len(features['news_columns'])} news features, "
        f"{features['sessions_without_news']} session(s) without news)"
    )
    for name in ("train", "validation", "test"):
        period = periods[name]
        print(
            f"{name:<10}: {period['rows']:>4} rows {period['first_session']}..{period['last_session']} "
            f"up-rate {period['positive_rate']:.3f}"
        )
    naive = report["models"]["naive_majority"]["test"]
    market = report["models"]["market_only"]["test"]
    with_news = report["models"]["market_plus_news"]["test"]
    print(f"naive    : acc {naive['accuracy']:.3f} bal-acc {naive['balanced_accuracy']:.3f} f1 {naive['f1_positive']:.3f}")
    print(
        f"market   : acc {market['accuracy']:.3f} bal-acc {market['balanced_accuracy']:.3f} "
        f"f1 {market['f1_positive']:.3f} roc-auc {market['roc_auc'] if market['roc_auc'] is not None else 'n/a'}"
    )
    print(
        f"+news    : acc {with_news['accuracy']:.3f} bal-acc {with_news['balanced_accuracy']:.3f} "
        f"f1 {with_news['f1_positive']:.3f} roc-auc {with_news['roc_auc'] if with_news['roc_auc'] is not None else 'n/a'}"
    )
    comparison = report["comparison"]
    print(
        f"delta    : {comparison['news_minus_market_only']:+.3f} balanced accuracy "
        f"(news beats market-only: {comparison['news_beats_market_only_on_test']})"
    )
    print(f"report   : {relative_to_root(results_dir(root) / REPORT_FILENAME, root)}")
    failed = [check for check in report["checks"] if not check["passed"]]
    for check in report["checks"]:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['name']}: {check['detail']}")
    if args.print_json:
        print(json.dumps(report, indent=2))
    if failed:
        print(f"\n{len(failed)} check(s) failed — the comparison is not reproducible as reported.")
        return 1
    print(f"\nAll {len(report['checks'])} Phase 4 checks passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
