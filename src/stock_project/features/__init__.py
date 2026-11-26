"""Market feature engineering (Phase 3).

``technical_indicators`` turns a cleaned market table into one feature row per
predicted session, using only bars strictly before that session.
``chronological_splits`` cuts those rows into ordered train/validation/test
periods without shuffling.

Import from the modules directly, e.g.
``from stock_project.features.technical_indicators import build_market_features``.
"""
