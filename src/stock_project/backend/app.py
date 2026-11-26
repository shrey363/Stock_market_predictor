"""Read-only Phase 7 Flask API over saved datasets and experiment artifacts.

Run with::

    python -m stock_project.backend.app --config configs/config.example.yaml

Requests never train a model or run a simulation. Every displayed number comes from
an existing CSV/JSON file so the dashboard stays equal to the experiment record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from ..common.config import Config, load_config
from ..common.paths import project_root, results_dir

MAX_ROWS = 10_000


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _inside_root(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"artifact path escapes the project root: {relative}")
    return path


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _page(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    try:
        limit = int(request.args.get("limit", 500))
        offset = int(request.args.get("offset", 0))
    except ValueError as error:
        raise ValueError("limit and offset must be integers") from error
    if not 1 <= limit <= MAX_ROWS or offset < 0:
        raise ValueError(f"limit must be 1..{MAX_ROWS} and offset must be non-negative")
    return frame.iloc[offset : offset + limit], {"total": len(frame), "limit": limit, "offset": offset}


def _report(root: Path, filename: str) -> dict[str, Any]:
    return _read_json(results_dir(root) / filename)


def create_app(config: Config | None = None, *, root: Path | str | None = None) -> Flask:
    """Create the read-only dashboard application, optionally rooted at a test directory."""
    base = project_root(root).resolve()
    config = config or load_config()
    frontend = base / "frontend"
    app = Flask(__name__, static_folder=None)

    @app.errorhandler(FileNotFoundError)
    def missing_artifact(error: FileNotFoundError):
        missing = Path(error.filename or str(error)).name
        return jsonify({"error": "artifact_missing", "detail": f"required saved artifact is missing: {missing}"}), 503

    @app.errorhandler(ValueError)
    def invalid_request(error: ValueError):
        return jsonify({"error": "invalid_request", "detail": str(error)}), 400

    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def dashboard():
        return send_from_directory(frontend, "index.html")

    @app.get("/market")
    def market():
        phase1 = _report(base, "phase1_data_report.json")
        phase4 = _report(base, "phase4_news_comparison_report.json")
        prices = _read_csv(_inside_root(base, phase1["datasets"]["market"]["processed_file"]))
        features = _read_csv(_inside_root(base, phase4["inputs"]["features_table"]))
        prices["session_date"] = pd.to_datetime(prices["timestamp"], utc=True).dt.strftime("%Y-%m-%d")
        indicator_columns = phase4["features"]["market_columns"]
        merged = prices.merge(
            features[["session_date", *indicator_columns]], on="session_date", how="left", validate="one_to_one"
        )
        page, pagination = _page(merged)
        return jsonify(
            {
                "instrument": config.market.symbol,
                "source": config.market.source,
                "indicators": indicator_columns,
                "pagination": pagination,
                "rows": _records(page),
            }
        )

    @app.get("/news")
    def news():
        phase2 = _report(base, "phase2_classification_report.json")
        frame = _read_csv(_inside_root(base, phase2["traceability"]["output_file"]))
        for field in ("event_category", "sentiment_label"):
            value = request.args.get(field)
            if value:
                frame = frame.loc[frame[field] == value]
        frame = frame.sort_values(["published_at", "article_id"], ascending=[False, True])
        page, pagination = _page(frame)
        columns = [
            "article_id",
            "published_at",
            "available_from",
            "headline",
            "source",
            "source_url",
            "event_category",
            "sentiment_label",
            "sentiment_score",
            "matched_keywords",
        ]
        return jsonify(
            {
                "method": phase2["method"],
                "category_distribution": phase2["classification"]["category_distribution"],
                "pagination": pagination,
                "rows": _records(page[columns]),
            }
        )

    @app.get("/predictions")
    def predictions():
        report = _report(base, "phase4_news_comparison_report.json")
        frame = _read_csv(results_dir(base) / "phase4_test_predictions.csv")
        page, pagination = _page(frame)
        coefficients = {
            name: sorted(
                model["standardised_coefficients"].items(), key=lambda item: abs(item[1]), reverse=True
            )[:10]
            for name, model in (
                ("market_only", report["models"]["market_only"]),
                ("market_plus_news", report["models"]["market_plus_news"]),
            )
        }
        return jsonify(
            {
                "period": report["splits"]["periods"]["test"],
                "cutoff": report["cutoff"],
                "top_standardised_coefficients": coefficients,
                "pagination": pagination,
                "rows": _records(page),
            }
        )

    @app.get("/models")
    def models():
        classification = _report(base, "phase2_classifier_evaluation.json")
        comparison = _report(base, "phase4_news_comparison_report.json")
        phase2 = classification["primary_sheet"]
        return jsonify(
            {
                "news_classifier": {
                    "test": phase2["metrics"],
                    "naive_accuracy": phase2["naive_majority_baseline"]["accuracy"],
                },
                "movement_models": {
                    "naive_majority": comparison["models"]["naive_majority"]["test"],
                    "market_only": comparison["models"]["market_only"]["test"],
                    "market_plus_news": comparison["models"]["market_plus_news"]["test"],
                },
                "comparison": comparison["comparison"],
            }
        )

    @app.get("/simulation")
    def simulation():
        report = _report(base, "phase6_rl_report.json")
        frame = _read_csv(results_dir(base) / "phase6_test_portfolio_values.csv")
        page, pagination = _page(frame)
        return jsonify(
            {
                "period": report["test"]["period"],
                "environment": report["environment"],
                "protocol": report["protocol"],
                "selection": report["selection"],
                "agent": report["test"]["agent"],
                "benchmarks": report["test"]["benchmarks"],
                "limitations": report["limitations"],
                "pagination": pagination,
                "rows": _records(page),
            }
        )

    @app.get("/methodology")
    def methodology():
        phase1 = _report(base, "phase1_data_report.json")
        phase2 = _report(base, "phase2_classifier_evaluation.json")
        phase4 = _report(base, "phase4_news_comparison_report.json")
        phase5 = _report(base, "phase5_simulation_report.json")
        phase6 = _report(base, "phase6_rl_report.json")
        return jsonify(
            {
                "disclaimer": "Research and simulated trading only; not financial advice or evidence of future profitability.",
                "data_sources": phase1["config"]["data"],
                "classifier_evaluation": phase2["primary_sheet"],
                "prediction_cutoff": phase4["cutoff"],
                "evaluation_period": phase4["splits"]["periods"]["test"],
                "execution_rules": phase5["execution_rules"],
                "rl_protocol": phase6["protocol"],
                "limitations": {
                    "models": phase4["limitations"],
                    "simulation": phase5["limitations"],
                    "reinforcement_learning": phase6["limitations"],
                },
            }
        )

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the read-only Phase 7 dashboard and API.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config")
    parser.add_argument("--root", type=Path, default=None, help="project root")
    parser.add_argument("--host", default="127.0.0.1", help="listen host (default: loopback only)")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("the research dashboard is local-only; use a loopback host")
    create_app(load_config(args.config), root=args.root).run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
