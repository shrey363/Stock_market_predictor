"""Phase 3 baseline: metrics on hand-checkable inputs and the artifact contract."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stock_project.common.config import Config, load_config
from stock_project.common.errors import DataNotFetchedError
from stock_project.data.inspect_data import market_processed_path
from stock_project.models.train_baseline import (
    PREDICTIONS_FILENAME,
    REPORT_FILENAME,
    classification_metrics,
    features_processed_path,
    main,
    run_baseline,
)

PREDICTION_COLUMNS = [
    "session_date",
    "decision_time_utc",
    "next_session_return",
    "target",
    "probability_up",
    "prediction",
    "correct",
]


def prepare_market_table(root: Path, config: Config, synthetic_market, rows: int = 220) -> Path:
    """Write a deterministic synthetic market table where the pipeline expects it."""
    target = market_processed_path(config, root=root)
    target.parent.mkdir(parents=True, exist_ok=True)
    synthetic_market(rows, seed=11).to_csv(target, index=False, lineterminator="\n")
    return target


def test_metrics_match_hand_computation():
    metrics = classification_metrics(pd.Series([1, 1, 0, 0]), pd.Series([1, 0, 0, 1]))
    assert metrics["rows"] == 4
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision_positive"] == pytest.approx(0.5)
    assert metrics["recall_positive"] == pytest.approx(0.5)
    assert metrics["f1_positive"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"]["rows_true_negative_col_predicted"] == [1, 1]
    assert metrics["confusion_matrix"]["rows_true_positive_col_predicted"] == [1, 1]
    assert metrics["predicted_positive"] == 2
    assert metrics["actual_positive"] == 2
    # One class absent in the predictions still yields a defined AUC; with no scores it is None.
    assert metrics["roc_auc"] is None


def test_auc_is_computed_when_scores_are_supplied():
    metrics = classification_metrics(
        pd.Series([1, 0, 1, 0]), pd.Series([1, 0, 1, 0]), pd.Series([0.9, 0.1, 0.8, 0.2])
    )
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_run_writes_artifacts_and_every_check_passes(tmp_path: Path, config_path: Path, synthetic_market):
    config = load_config(config_path)
    prepare_market_table(tmp_path, config, synthetic_market)

    report = run_baseline(config, root=tmp_path)

    report_file = tmp_path / "reports" / "experiment_results" / REPORT_FILENAME
    on_disk = json.loads(report_file.read_text(encoding="utf-8"))
    assert on_disk["phase"] == 3
    assert on_disk["generated_at_utc"].endswith("Z")
    assert not [check for check in on_disk["checks"] if not check["passed"]], on_disk["checks"]
    assert len(on_disk["checks"]) == 8
    assert on_disk["next_step"].startswith("Phase 4")

    # Every market session is either a feature row or a counted warm-up row.
    assert on_disk["target_balance"]["rows"] + on_disk["features"]["warmup_rows_dropped"] == 220
    assert on_disk["features"]["columns"] == list(on_disk["features"]["definitions"])
    assert on_disk["features"]["columns"] and not set(on_disk["features"]["columns"]) & {"target", "next_session_return"}

    periods = on_disk["splits"]["periods"]
    assert periods["train"]["last_session"] < periods["validation"]["first_session"]
    assert periods["validation"]["last_session"] < periods["test"]["first_session"]
    ranges = on_disk["splits"]["ranges"]
    assert ranges["train"][0] == 0 and ranges["test"][1] == on_disk["target_balance"]["rows"]
    assert (
        periods["train"]["rows"] + periods["validation"]["rows"] + periods["test"]["rows"]
        == on_disk["target_balance"]["rows"]
    )

    market_only = on_disk["models"]["market_only"]
    assert market_only["selected_c"] in config.model.c_values
    assert market_only["fit_rows"] == periods["train"]["rows"] + periods["validation"]["rows"]
    assert len(market_only["selection_on_validation"]) == len(config.model.c_values)
    assert on_disk["comparison"]["market_only"] == market_only["test"]["balanced_accuracy"]
    assert on_disk["models"]["naive_majority"]["predicted_class"] in (0, 1)
    for key in ("accuracy", "balanced_accuracy", "precision_positive", "recall_positive", "f1_positive", "confusion_matrix"):
        assert key in market_only["test"]
    assert len(market_only["standardised_coefficients"]) == len(on_disk["features"]["columns"])

    features_file = features_processed_path(config, root=tmp_path)
    predictions_file = tmp_path / "reports" / "experiment_results" / PREDICTIONS_FILENAME
    assert features_file.exists() and predictions_file.exists()

    features = pd.read_csv(features_file)
    assert len(features) == on_disk["target_balance"]["rows"]
    predictions = pd.read_csv(predictions_file)
    assert list(predictions.columns) == PREDICTION_COLUMNS
    assert len(predictions) == periods["test"]["rows"]
    assert predictions["session_date"].is_monotonic_increasing
    assert predictions["probability_up"].between(0.0, 1.0).all()
    assert set(predictions["prediction"].unique()) <= {0, 1}
    assert report["models"]["market_only"]["test"]["rows"] == len(predictions)

    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["datasets"]["processed:features:SPY:1d"]
    assert entry["rows"] == len(features)
    assert any("invariant 2" in note for note in entry["notes"])


def test_run_is_deterministic(tmp_path: Path, config_path: Path, synthetic_market):
    config = load_config(config_path)
    prepare_market_table(tmp_path, config, synthetic_market)
    first = run_baseline(config, root=tmp_path, write=False)
    second = run_baseline(config, root=tmp_path, write=False)
    assert first["models"] == second["models"]
    assert first["comparison"] == second["comparison"]
    assert first["splits"] == second["splits"]


def test_missing_market_table_points_at_phase_one(tmp_path: Path, config_path: Path):
    config = load_config(config_path)
    with pytest.raises(DataNotFetchedError, match="inspect_data"):
        run_baseline(config, root=tmp_path)


def test_cli_exits_zero_and_prints_the_comparison(tmp_path: Path, config_path: Path, synthetic_market, capsys):
    config = load_config(config_path)
    prepare_market_table(tmp_path, config, synthetic_market)

    exit_code = main(["--config", str(config_path), "--root", str(tmp_path)])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "All 8 Phase 3 checks passed." in captured
    assert "naive    :" in captured and "model    :" in captured
    assert "selected : C=" in captured
    assert (tmp_path / "reports" / "experiment_results" / REPORT_FILENAME).exists()
