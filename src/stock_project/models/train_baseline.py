"""CLI: train the Phase 3 naive baseline and market-only model.

    python -m stock_project.models.train_baseline --config configs/config.example.yaml

Reads the cleaned market table produced by Phase 1, builds one feature row per
session (cutoff: strictly before that session), cuts the rows chronologically into
train / validation / test, selects the regularisation strength on validation only,
refits on train + validation, and evaluates once on the held-out test period.

Artifacts (invariant 7 — the dashboard reads files, never recomputes):

* ``data/processed/features_<SYMBOL>_<interval>.csv`` — the feature table, also
  recorded in ``data/manifest.json``.
* ``reports/experiment_results/phase3_baseline_report.json`` — config snapshot,
  feature definitions, split summaries, metrics for the naive baseline and the
  model on validation and test, coefficients, checks and limitations.
* ``reports/experiment_results/phase3_test_predictions.csv`` — one row per test
  session, so any reported number can be traced back to a date and a probability.

The naive baseline is the majority class of the *training* period. A model that
cannot beat it is a valid, reportable Phase 3 outcome — this is a baseline study,
not a profit claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..common.config import Config, load_config
from ..common.errors import DataNotFetchedError
from ..common.logging_utils import configure_cli_logging, get_logger
from ..common.manifest import make_dataset_entry, write_manifest
from ..common.paths import ensure_directories, processed_dir, relative_to_root, results_dir
from ..common.timeutils import utc_now_iso
from ..data.inspect_data import market_processed_path
from ..features.chronological_splits import chronological_split, train_validation_test
from ..features.technical_indicators import (
    CONTEXT_COLUMNS,
    TARGET_COLUMNS,
    build_market_features,
    feature_definitions,
)

logger = get_logger("stock_project.models.train_baseline")

REPORT_FILENAME = "phase3_baseline_report.json"
PREDICTIONS_FILENAME = "phase3_test_predictions.csv"

#: Metric used to compare the naive baseline with the model, in the report.
COMPARISON_METRIC = "balanced_accuracy"


def features_processed_path(config: Config, *, root: Path | None = None) -> Path:
    return processed_dir(root) / f"features_{config.market.symbol.upper()}_{config.market.interval}.csv"


def classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_score: pd.Series | None = None,
) -> dict[str, Any]:
    """Accuracy plus the imbalance-aware metrics the specification asks for.

    ``precision``/``recall``/``f1`` describe the positive ("session closes up")
    class, which is the class the trading question cares about; ``roc_auc`` is
    reported only when both classes are present, otherwise it would be undefined.
    """
    truth = pd.Series(y_true).astype("int64").reset_index(drop=True)
    prediction = pd.Series(y_pred).astype("int64").reset_index(drop=True)
    labels = [0, 1]
    matrix = confusion_matrix(truth, prediction, labels=labels)
    metrics: dict[str, Any] = {
        "rows": int(len(truth)),
        "accuracy": round(float(accuracy_score(truth, prediction)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(truth, prediction)), 6),
        "precision_positive": round(float(precision_score(truth, prediction, pos_label=1, zero_division=0)), 6),
        "recall_positive": round(float(recall_score(truth, prediction, pos_label=1, zero_division=0)), 6),
        "f1_positive": round(float(f1_score(truth, prediction, pos_label=1, zero_division=0)), 6),
        "confusion_matrix": {
            "labels": labels,
            "rows_true_negative_col_predicted": [int(value) for value in matrix[0]],
            "rows_true_positive_col_predicted": [int(value) for value in matrix[1]],
        },
        "predicted_positive": int(prediction.sum()),
        "actual_positive": int(truth.sum()),
    }
    if y_score is not None and truth.nunique() == 2:
        metrics["roc_auc"] = round(float(roc_auc_score(truth, pd.Series(y_score))), 6)
    else:
        metrics["roc_auc"] = None
    return metrics


def _majority_class(train_target: pd.Series) -> int:
    """Most frequent training label; a tie is broken toward the positive class."""
    counts = train_target.value_counts()
    return 1 if int(counts.get(1, 0)) >= int(counts.get(0, 0)) else 0


def _fit_logistic(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    *,
    c: float,
    max_iter: int,
    seed: int,
) -> Pipeline:
    """StandardScaler + LogisticRegression, both fit on the supplied rows only."""
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(C=c, max_iter=max_iter, solver="lbfgs", random_state=seed),
            ),
        ]
    )
    pipeline.fit(train_features, train_target)
    return pipeline


def _probabilities(pipeline: Pipeline, features: pd.DataFrame) -> pd.Series:
    return pd.Series(pipeline.predict_proba(features)[:, 1], index=features.index)


def run_baseline(config: Config, *, root: Path | None = None, write: bool = True) -> dict[str, Any]:
    """Build features, split chronologically, train both baselines, return the report."""
    market_file = market_processed_path(config, root=root)
    if not market_file.exists():
        raise DataNotFetchedError(
            f"no cleaned market table at {market_file}; run: "
            "python -m stock_project.data.inspect_data --config configs/config.example.yaml"
        )

    market = pd.read_csv(market_file, dtype={"symbol": "string"})
    definitions = feature_definitions(
        return_windows=config.features.return_windows,
        moving_average_windows=config.features.moving_average_windows,
        rsi_window=config.features.rsi_window,
        volatility_window=config.features.volatility_window,
        volume_window=config.features.volume_window,
    )
    features, feature_quality = build_market_features(
        market,
        return_windows=config.features.return_windows,
        moving_average_windows=config.features.moving_average_windows,
        rsi_window=config.features.rsi_window,
        volatility_window=config.features.volatility_window,
        volume_window=config.features.volume_window,
        cutoff_time=config.prediction.cutoff_time,
    )
    feature_columns = list(definitions)
    absent = [column for column in feature_columns if column not in features.columns]
    if absent:  # pragma: no cover - defensive contract check
        raise AssertionError(f"feature definitions name columns the table lacks: {absent}")

    split = chronological_split(
        len(features),
        train_fraction=config.splits.train_fraction,
        validation_fraction=config.splits.validation_fraction,
    )
    split_summaries = train_validation_test(features, split)
    split_summaries["test"]["share_of_rows"] = round(len(split.test) / len(features), 6)

    train_features, train_target = features.iloc[split.train.start : split.train.stop][feature_columns], features.iloc[split.train.start : split.train.stop]["target"]
    validation_features = features.iloc[split.validation.start : split.validation.stop][feature_columns]
    validation_target = features.iloc[split.validation.start : split.validation.stop]["target"]
    test_features = features.iloc[split.test.start : split.test.stop][feature_columns]
    test_target = features.iloc[split.test.start : split.test.stop]["target"]

    majority = _majority_class(train_target)
    naive_test = classification_metrics(test_target, pd.Series([majority] * len(test_target)))

    selection: list[dict[str, Any]] = []
    best_c = None
    best_score = None
    for c in config.model.c_values:
        candidate = _fit_logistic(train_features, train_target, c=c, max_iter=config.model.max_iter, seed=config.project.random_seed)
        candidate_metrics = classification_metrics(
            validation_target, candidate.predict(validation_features), _probabilities(candidate, validation_features)
        )
        score = candidate_metrics[config.model.selection_metric]
        selection.append({"c": float(c), "validation": candidate_metrics})
        # Strict improvement only: on a tie the earlier (smaller) C wins, deterministically.
        if best_score is None or score > best_score + 1e-12:
            best_score, best_c = score, float(c)
    if best_c is None:  # pragma: no cover - config validation prevents an empty grid
        raise AssertionError("no regularisation strength was evaluated")

    # Refit on train + validation (the selected setting is frozen), then touch the
    # held-out period exactly once.
    fit_features = pd.concat([train_features, validation_features])
    fit_target = pd.concat([train_target, validation_target])
    final_model = _fit_logistic(fit_features, fit_target, c=best_c, max_iter=config.model.max_iter, seed=config.project.random_seed)

    test_probability = _probabilities(final_model, test_features)
    test_prediction = pd.Series(final_model.predict(test_features), index=test_features.index)
    model_test = classification_metrics(test_target, test_prediction, test_probability)
    chosen_validation = next(entry["validation"] for entry in selection if entry["c"] == best_c)

    coefficients = {
        column: round(float(value), 6)
        for column, value in zip(feature_columns, final_model.named_steps["model"].coef_[0])
    }
    coefficients_sorted = dict(sorted(coefficients.items(), key=lambda item: abs(item[1]), reverse=True))

    checks = [
        {
            "name": "market_rows_loaded",
            "passed": len(market) > 0,
            "detail": f"{len(market)} cleaned session(s) read from {relative_to_root(market_file, root)}",
        },
        {
            "name": "feature_rows_present",
            "passed": len(features) > 0,
            "detail": f"{len(features)} feature row(s) after dropping {feature_quality.rows_dropped_missing_required} warm-up row(s)",
        },
        {
            "name": "features_chronological",
            "passed": bool(features["session_date"].is_monotonic_increasing),
            "detail": "feature rows are in ascending session order; nothing was shuffled",
        },
        {
            "name": "features_no_missing_values",
            "passed": not bool(features[feature_columns].isna().any().any()),
            "detail": "every kept row has a value for every feature; warm-up rows were dropped, not filled",
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
            "passed": bool(
                ((features["next_session_return"] > 0).astype("int64") == features["target"]).all()
            ),
            "detail": "target is 1 exactly when the predicted session's return is above zero",
        },
        {
            "name": "features_exclude_target",
            "passed": not set(feature_columns) & (set(CONTEXT_COLUMNS) | set(TARGET_COLUMNS)),
            "detail": "no context or target column is used as a model input",
        },
        {
            "name": "model_fit_rows_exclude_test",
            "passed": len(fit_features) == len(split.train) + len(split.validation),
            "detail": f"the final model was fit on {len(fit_features)} train+validation rows; the {len(split.test)} test rows were not used",
        },
    ]

    report: dict[str, Any] = {
        "generated_at_utc": utc_now_iso(),
        "phase": 3,
        "phase_name": "Market features and baseline prediction",
        "config": config.to_dict(),
        "inputs": {
            "market_table": relative_to_root(market_file, root),
            "market_rows": int(len(market)),
            "features_table": relative_to_root(features_processed_path(config, root=root), root),
        },
        "cutoff": {
            "decision_rule": (
                "A row for session t is decided at that session's timestamp (pre-open, 05:00 UTC in this "
                "series) and every feature uses bars up to and including t-1 only; values are computed at "
                "the close of their bar and shifted forward one row."
            ),
            "news_cutoff_rule": (
                f"news_cutoff_utc = session date at {config.prediction.cutoff_time} UTC. Phase 4 may only "
                "use articles whose available_from is at or before it, i.e. day-precision news published on "
                "the previous day at the latest. Phase 3 reads no news at all."
            ),
            "target": (
                "target = 1 when close[t] > close[t-1] for the predicted session t; next_session_return "
                "records the exact return it was derived from. No purge gap is needed because no row depends "
                "on another row's period."
            ),
        },
        "features": {
            "columns": feature_columns,
            "definitions": definitions,
            "warmup_rows_dropped": int(feature_quality.rows_dropped_missing_required),
            "notes": list(feature_quality.notes),
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
            "market_only": {
                "type": config.model.type,
                "inputs": "market features only (no news)",
                "selection_metric": config.model.selection_metric,
                "selection_on_validation": selection,
                "selected_c": best_c,
                "fit_rows": int(len(fit_features)),
                "validation_of_selected_c": chosen_validation,
                "test": model_test,
                "standardised_coefficients": coefficients_sorted,
            },
        },
        "comparison": {
            "metric": COMPARISON_METRIC,
            "naive_majority": naive_test[COMPARISON_METRIC],
            "market_only": model_test[COMPARISON_METRIC],
            "beats_naive_on_test": bool(model_test[COMPARISON_METRIC] > naive_test[COMPARISON_METRIC]),
            "note": (
                "A single held-out period of a few hundred sessions is a small sample: a difference of a few "
                "points is indistinguishable from noise. This result is a baseline measurement, not evidence of "
                "predictive skill or profitability."
            ),
        },
        "checks": checks,
        "limitations": [
            "One instrument (SPY) and one historical window: results do not generalise to other markets or periods.",
            "Daily bars only; no intraday information, order flow, or after-hours prices.",
            "Precision/recall/F1 describe the positive ('closes up') class; SPY's positive base rate is near 55%, so accuracy alone is misleading — balanced accuracy and the naive baseline are the fair comparison.",
            "Hyper-parameter selection used a single validation period; with this few rows the selected C is itself noisy.",
            "No news features are used yet (Phase 4); this is the market-only baseline the core research question compares against.",
            "Transaction costs and execution are irrelevant here because no trading is simulated (Phase 5).",
            "This is a research baseline, not financial advice, and it makes no claim of reliable prediction.",
        ],
        "next_step": (
            "Phase 4 — aggregate pre-cutoff news into per-session features (counts by category, sentiment, "
            "no-news indicator) and train the same model family on the same splits to answer the research question."
        ),
    }

    if write:
        ensure_directories(root)
        feature_out = features_processed_path(config, root=root)
        features.to_csv(feature_out, index=False, lineterminator="\n")
        write_manifest(
            root,
            key=f"processed:features:{config.market.symbol.upper()}:{config.market.interval}",
            entry=make_dataset_entry(
                key=f"processed:features:{config.market.symbol.upper()}:{config.market.interval}",
                name=f"market features and next-session target ({config.market.symbol.upper()} {config.market.interval})",
                file_path=feature_out,
                root=root,
                source=relative_to_root(market_file, root),
                rows=int(len(features)),
                columns=list(features.columns),
                date_coverage=(str(features["session_date"].iloc[0]), str(features["session_date"].iloc[-1])),
                notes=[
                    "Derived by features.technical_indicators from the cleaned market table.",
                    f"{feature_quality.rows_dropped_missing_required} warm-up row(s) dropped; no imputation.",
                    "Features use bars up to and including t-1 for the session t row (invariant 2).",
                ],
            ),
        )

        predictions = pd.DataFrame(
            {
                "session_date": features.iloc[split.test.start : split.test.stop]["session_date"].to_numpy(),
                "decision_time_utc": features.iloc[split.test.start : split.test.stop]["decision_time_utc"].to_numpy(),
                "next_session_return": features.iloc[split.test.start : split.test.stop]["next_session_return"].to_numpy(),
                "target": test_target.to_numpy(),
                "probability_up": test_probability.to_numpy(),
                "prediction": test_prediction.to_numpy(),
            }
        )
        predictions["correct"] = (predictions["prediction"] == predictions["target"]).astype(int)
        predictions_path = results_dir(root) / PREDICTIONS_FILENAME
        predictions.to_csv(predictions_path, index=False, lineterminator="\n")

        report_path = results_dir(root) / REPORT_FILENAME
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logger.info("wrote %s", relative_to_root(report_path, root))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build market features and train the Phase 3 naive and market-only baselines."
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config (default: configs/config.yaml or the example)")
    parser.add_argument("--print-json", action="store_true", help="also print the full report to stdout")
    parser.add_argument("--root", type=Path, default=None, help="project root to read/write (default: this repository)")
    args = parser.parse_args(argv)

    configure_cli_logging()
    config = load_config(args.config)
    root = Path(args.root) if args.root else None
    report = run_baseline(config, root=root)

    periods = report["splits"]["periods"]
    print(
        f"features : {report['target_balance']['rows']} rows "
        f"{periods['train']['first_session']}..{periods['test']['last_session']} "
        f"({len(report['features']['columns'])} features, {report['features']['warmup_rows_dropped']} warm-up rows dropped)"
    )
    for name in ("train", "validation", "test"):
        period = periods[name]
        print(
            f"{name:<10}: {period['rows']:>4} rows {period['first_session']}..{period['last_session']} "
            f"up-rate {period['positive_rate']:.3f}"
        )
    naive = report["models"]["naive_majority"]["test"]
    model = report["models"]["market_only"]["test"]
    print(f"split    : {report['splits']['ranges']}")
    print(f"selected : C={report['models']['market_only']['selected_c']} on validation {report['models']['market_only']['selection_metric']}")
    print(f"naive    : acc {naive['accuracy']:.3f} bal-acc {naive['balanced_accuracy']:.3f} f1 {naive['f1_positive']:.3f}")
    print(
        f"model    : acc {model['accuracy']:.3f} bal-acc {model['balanced_accuracy']:.3f} "
        f"f1 {model['f1_positive']:.3f} roc-auc {model['roc_auc'] if model['roc_auc'] is not None else 'n/a'}"
    )
    print(f"report   : {relative_to_root(results_dir(root) / REPORT_FILENAME, root)}")
    failed = [check for check in report["checks"] if not check["passed"]]
    for check in report["checks"]:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['name']}: {check['detail']}")
    if args.print_json:
        print(json.dumps(report, indent=2))
    if failed:
        print(f"\n{len(failed)} check(s) failed — the baseline is not reproducible as reported.")
        return 1
    print(f"\nAll {len(report['checks'])} Phase 3 checks passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
