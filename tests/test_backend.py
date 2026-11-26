"""Phase 7 read-only API and dashboard-to-artifact contract."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess

import pandas as pd
import pytest

pytest.importorskip("flask")

from stock_project.backend.app import create_app
from stock_project.common.config import load_config


def write_json(root: Path, name: str, value: dict) -> None:
    path = root / "reports" / "experiment_results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(root: Path, relative: str, rows: list[dict]) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def phase7_root(tmp_path: Path, config_path: Path) -> Path:
    market_file = "data/processed/market.csv"
    news_file = "data/processed/news.csv"
    features_file = "data/processed/features.csv"
    write_csv(tmp_path, market_file, [{"symbol": "SPY", "timestamp": "2022-01-03T05:00:00Z", "open": 10, "high": 12, "low": 9, "close": 11, "adj_close": 11, "volume": 100, "source": "fixture"}])
    write_csv(tmp_path, features_file, [{"session_date": "2022-01-03", "return_1d": 0.1}])
    write_csv(tmp_path, news_file, [{"article_id": "a", "published_at": "2022-01-02T00:00:00Z", "available_from": "2022-01-03T00:00:00Z", "headline": "Headline", "source": "fixture", "source_url": "https://example.test/a", "event_category": "sanctions", "sentiment_label": "neutral", "sentiment_score": 0.0, "matched_keywords": "sanctions"}])
    write_csv(tmp_path, "reports/experiment_results/phase4_test_predictions.csv", [{"session_date": "2022-01-03", "target": 1, "probability_up_market_only": 0.4, "probability_up_market_news": 0.6, "correct_market_only": 0, "correct_market_news": 1}])
    equity = {"strategy": "ppo_seed_0", "seed": 0, "session_date": "2022-01-03", "action_name": "hold", "executed": False, "transaction_cost": 0.0, "cash": 1000.0, "shares_after": 0, "portfolio_value": 1000.0}
    write_csv(tmp_path, "reports/experiment_results/phase6_test_portfolio_values.csv", [equity])

    write_json(tmp_path, "phase1_data_report.json", {"config": {"data": {"market": {}, "news": {}}}, "datasets": {"market": {"processed_file": market_file}}})
    write_json(tmp_path, "phase2_classification_report.json", {"method": {"event_categories": ["sanctions"]}, "classification": {"category_distribution": {"sanctions": {"count": 1, "share": 1.0}}}, "traceability": {"output_file": news_file}})
    metrics = {"accuracy": 0.9, "balanced_accuracy": 0.4, "cohen_kappa": 0.2, "macro_average": {"f1": 0.3}}
    write_json(tmp_path, "phase2_classifier_evaluation.json", {"primary_sheet": {"metrics": metrics, "naive_majority_baseline": {"accuracy": 0.9}}})
    model = {"test": {"balanced_accuracy": 0.5}, "standardised_coefficients": {"return_1d": 0.1}}
    write_json(tmp_path, "phase4_news_comparison_report.json", {"inputs": {"features_table": features_file}, "features": {"market_columns": ["return_1d"]}, "cutoff": {"news_rule": "pre-cutoff only"}, "splits": {"periods": {"test": {"first_session": "2022-01-03", "last_session": "2022-01-03"}}}, "models": {"naive_majority": {"test": {"balanced_accuracy": 0.5}}, "market_only": model, "market_plus_news": model}, "comparison": {"news_minus_market_only": 0.0, "note": "no difference"}, "limitations": ["small sample"]})
    write_json(tmp_path, "phase5_simulation_report.json", {"execution_rules": {"timing": "open to close"}, "limitations": ["historical"]})
    strategy = {"strategy": "ppo_seed_0", "sessions": 1, "final_value": 1000.0, "total_return": 0.0, "number_of_trades": 0, "total_transaction_costs": 0.0, "max_drawdown": 0.0, "sharpe_ratio": None, "seed": 0}
    write_json(tmp_path, "phase6_rl_report.json", {"environment": {"actions": {"hold": 0}}, "protocol": {"seeds": [0]}, "selection": {"selected": {"name": "ppo"}}, "test": {"period": {"sessions": 1, "first_session": "2022-01-03", "last_session": "2022-01-03"}, "agent": {"per_seed": [strategy], "aggregate": {"mean_total_return": 0.0}}, "benchmarks": {}}, "limitations": ["stochastic"]})
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    frontend.joinpath("index.html").write_text("<!doctype html><title>Dashboard fixture</title>", encoding="utf-8")
    return tmp_path


@pytest.fixture
def client(phase7_root: Path, config_path: Path):
    app = create_app(load_config(config_path), root=phase7_root)
    app.config.update(TESTING=True)
    return app.test_client()


def test_all_api_endpoints_are_read_only_and_serve_saved_values(client):
    expected = {
        "/market?limit=1": ("rows", "close", 11),
        "/news?limit=1": ("rows", "headline", "Headline"),
        "/predictions?limit=1": ("rows", "probability_up_market_news", 0.6),
        "/models": ("movement_models", "market_plus_news", {"balanced_accuracy": 0.5}),
        "/simulation?limit=1": ("rows", "portfolio_value", 1000.0),
        "/methodology": ("evaluation_period", "first_session", "2022-01-03"),
    }
    for path, (group, key, value) in expected.items():
        response = client.get(path)
        assert response.status_code == 200, path
        payload = response.get_json()
        actual = payload[group][0][key] if isinstance(payload[group], list) else payload[group][key]
        assert actual == value
        assert client.post(path).status_code == 405


def test_news_filters_and_pagination_are_bounded(client):
    response = client.get("/news?event_category=sanctions&limit=1&offset=0")
    assert response.get_json()["pagination"] == {"total": 1, "limit": 1, "offset": 0}
    assert client.get("/news?limit=0").status_code == 400
    assert client.get("/news?limit=10001").status_code == 400
    assert client.get("/news?offset=-1").status_code == 400


def test_missing_artifact_is_an_actionable_service_error(client, phase7_root: Path):
    (phase7_root / "reports" / "experiment_results" / "phase6_rl_report.json").unlink()
    response = client.get("/simulation")
    assert response.status_code == 503
    assert response.get_json()["error"] == "artifact_missing"
    assert "phase6_rl_report.json" in response.get_json()["detail"]


def test_dashboard_is_served_and_names_every_required_section(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert b"Dashboard fixture" in response.data


def test_real_dashboard_reads_api_routes_and_contains_responsible_use_copy():
    page = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")
    for route in ("/market", "/news", "/predictions", "/models", "/simulation", "/methodology"):
        assert route in page
    for section in ("Market overview", "News explorer", "Prediction view", "Model comparison", "Trading simulation", "Methodology & limitations"):
        assert section in page
    assert "not a trading signal" in page
    assert "Accessible table view" in page
    assert "ReactDOM.createRoot" in page and "Plotly.react" in page
    assert 'rel="icon"' in page


def test_inline_dashboard_javascript_parses_when_node_is_available():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is optional; browser assets have no local build step")
    page = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")
    scripts = [source for source in re.findall(r"<script(?: [^>]*)?>([\s\S]*?)</script>", page) if source.strip()]
    for source in scripts:
        result = subprocess.run(
            [node, "--check", "-"], input=source.encode("utf-8"), capture_output=True, check=False
        )
        assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
