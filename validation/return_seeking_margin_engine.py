#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

try:
    import yfinance as yf
except ModuleNotFoundError:
    yf = None

try:
    from validation.margin_strategy_validation import (
        BacktestConfig,
        MarginState,
        StateMachineConfig,
        generate_dynamic_targets,
        json_safe,
        load_config as load_existing_config,
        load_score_history,
        metrics as account_metrics,
        save_json,
        signal_from_history_row,
        simulate_account,
        update_margin_state,
    )
except ImportError:
    from margin_strategy_validation import (
        BacktestConfig,
        MarginState,
        StateMachineConfig,
        generate_dynamic_targets,
        json_safe,
        load_config as load_existing_config,
        load_score_history,
        metrics as account_metrics,
        save_json,
        signal_from_history_row,
        simulate_account,
        update_margin_state,
    )


@dataclass
class ReturnAlphaConfig:
    model_version: str = "v3_rank_relative_alpha_nonoverlap"
    ticker: str = "QQQ"
    horizons: Tuple[int, ...] = (20, 60, 120)
    horizon_weights: Tuple[float, ...] = (0.20, 0.55, 0.25)
    min_train_days: int = 1260
    train_window_days: int = 2520
    refit_frequency: str = "monthly"
    ridge_penalty: float = 50.0
    target_winsor_lower: float = 0.01
    target_winsor_upper: float = 0.99
    annual_borrow_rate: float = 0.04
    round_trip_cost_bps: float = 14.0

    # Lower-overlap training and direct relative-alpha target.
    training_stride_days: int = 10
    recency_half_life_days: int = 756
    static_baseline_window_labels: int = 1000
    static_baseline_min_samples: int = 80
    expert_min_train_rows: int = 300

    # Non-overlapping decision cadence and rank stability.
    decision_interval_business_days: int = 10
    rank_stability_lookback_days: int = 5
    rank_stability_min_observations: int = 3
    rank_stability_min_passes: int = 2

    # Rank-based sparse allocation. Probability is diagnostic only.
    rank_tier_1: float = 0.75
    rank_tier_2: float = 0.85
    rank_tier_3: float = 0.92
    rank_tier_4: float = 0.97
    relative_alpha_tier_1_pct: float = 0.0
    relative_alpha_tier_2_pct: float = 0.50
    relative_alpha_tier_3_pct: float = 1.00
    relative_alpha_tier_4_pct: float = 2.00
    margin_tier_1_pct: float = 2.0
    margin_tier_2_pct: float = 4.0
    margin_tier_3_pct: float = 6.0
    margin_tier_4_pct: float = 8.0
    max_alpha_margin_pct: float = 8.0
    require_price_above_ma200: bool = True
    risk_off_margin_pct: float = 0.0

    # Risk cap remains separate from Alpha Demand.
    backwardation_cap_pct: float = 2.0
    risk_cap_margin_45_50: float = 2.0
    risk_cap_margin_50_55: float = 4.0
    risk_cap_margin_55_60: float = 6.0
    risk_cap_margin_60_70: float = 8.0
    risk_cap_margin_70_plus: float = 10.0
    risk_penalty_if_risk_above_55: float = 2.0
    risk_penalty_if_liquidity_below_45: float = 1.0
    coverage_cap_if_below_80: float = 4.0
    regime_cap_if_below_45: float = 2.0

    # Daily invalidation around a slower 10-business-day entry schedule.
    exit_rank_percentile: float = 0.55
    exit_relative_alpha_pct: float = 0.0
    time_stop_business_days: int = 20
    time_stop_min_return_pct: float = 0.0
    breadth_shock_5d_threshold: float = -12.0
    risk_score_shock_5d_threshold: float = 10.0
    shock_reduce_fraction: float = 0.50

    # Validation.
    evaluation_fold_years: int = 2
    bootstrap_iterations: int = 2000
    bootstrap_block_months: int = 3
    random_seed: int = 415

    # Sensitivity neighborhood. A dedicated config may override these.
    sensitivity_rank_offsets: Tuple[float, ...] = (-0.03, 0.0, 0.03)
    sensitivity_alpha_offsets_pct: Tuple[float, ...] = (-0.25, 0.0, 0.25)
    sensitivity_decision_intervals: Tuple[int, ...] = (5, 10, 20)
    sensitivity_max_margins: Tuple[float, ...] = (6.0, 8.0)


@dataclass
class ReturnAlphaState:
    model_actual_margin_pct: float = 0.0
    confirmed_target_margin_pct: float = 0.0
    increase_streak: int = 0
    decrease_streak: int = 0
    last_trade_date: Optional[str] = None
    last_increase_date: Optional[str] = None
    last_increase_iso_week: Optional[str] = None
    last_processed_signal_date: Optional[str] = None
    entry_date: Optional[str] = None
    entry_price: Optional[float] = None
    peak_price: Optional[float] = None
    last_model_decision_date: Optional[str] = None
    scheduled_alpha_demand_margin_pct: float = 0.0
    scheduled_forecast_rank: Optional[float] = None
    scheduled_expected_relative_alpha_pct: Optional[float] = None
    last_action: str = "INITIALIZE"
    last_reason: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.last_reason is None:
            self.last_reason = []

FEATURE_BASE_COLUMNS = [
    "Market_Regime",
    "AI_Cycle",
    "Valuation",
    "Macro",
    "Liquidity",
    "Positioning",
    "Buy_Score",
    "Risk_Score",
    "Margin_Score",
    "Coverage_Ratio",
    "HY_OAS_Percentile",
    "Breadth_Score",
    "VIX",
    "VIX3M",
    "VIX_Backwardation",
    "NFL_13W_Change",
    "Net_Fed_Liquidity",
]


def load_dataclass_config(path: Optional[str], cls: Any) -> Any:
    instance = cls()
    if not path:
        return instance
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {unknown}")
    values = asdict(instance)
    values.update(payload)
    if cls is ReturnAlphaConfig:
        for name in (
            "horizons",
            "horizon_weights",
            "sensitivity_rank_offsets",
            "sensitivity_alpha_offsets_pct",
            "sensitivity_decision_intervals",
            "sensitivity_max_margins",
        ):
            if name in values:
                values[name] = tuple(values[name])
    return cls(**values)



def load_margin_state(path: Path, initial_actual_margin_pct: float = 0.0) -> ReturnAlphaState:
    if not path.exists():
        return ReturnAlphaState(
            model_actual_margin_pct=float(initial_actual_margin_pct),
            confirmed_target_margin_pct=float(initial_actual_margin_pct),
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(ReturnAlphaState)}
    clean = {key: value for key, value in payload.items() if key in allowed}
    return ReturnAlphaState(**clean)

def load_price_series(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    price_csv: Optional[str],
) -> pd.Series:
    if price_csv:
        frame = pd.read_csv(price_csv)
        date_col = next((c for c in ("date", "Date") if c in frame.columns), None)
        price_col = next(
            (c for c in ("price", "Adj Close", "adj_close", "Close", "close") if c in frame.columns),
            None,
        )
        if date_col is None or price_col is None:
            raise ValueError("Price CSV requires date/Date and price/Adj Close/Close.")
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame[price_col] = pd.to_numeric(frame[price_col], errors="coerce")
        series = (
            frame.dropna(subset=[date_col, price_col])
            .drop_duplicates(date_col, keep="last")
            .set_index(date_col)[price_col]
            .sort_index()
        )
    else:
        if yf is None:
            raise RuntimeError("yfinance is required when --price-csv is not provided.")
        raw = yf.download(
            ticker,
            start=(pd.Timestamp(start) - pd.Timedelta(days=400)).date(),
            end=(pd.Timestamp(end) + pd.Timedelta(days=10)).date(),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if raw.empty:
            raise RuntimeError(f"No price data returned for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            close_cols = [c for c in raw.columns if c[0] == "Close"]
            if not close_cols:
                raise RuntimeError("Downloaded price data has no Close column.")
            series = raw[close_cols[0]]
        else:
            series = raw["Close"]
    series.index = pd.to_datetime(series.index).tz_localize(None)
    return pd.to_numeric(series, errors="coerce").dropna().sort_index()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def build_features(history: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    common = history.index.intersection(prices.index)
    if len(common) < 500:
        raise ValueError("Insufficient common history and price dates.")
    history = history.reindex(common).copy()
    price = prices.reindex(common).astype(float)
    returns = price.pct_change()

    feature = pd.DataFrame(index=common)
    for column in FEATURE_BASE_COLUMNS:
        if column in history.columns:
            feature[column] = pd.to_numeric(history[column], errors="coerce")

    for days in (5, 10, 20, 60, 120, 252):
        feature[f"momentum_{days}d"] = price.pct_change(days)

    for days in (20, 50, 100, 200):
        ma = price.rolling(days, min_periods=days).mean()
        feature[f"price_to_ma{days}"] = price / ma - 1.0

    ma50 = price.rolling(50, min_periods=50).mean()
    ma200 = price.rolling(200, min_periods=200).mean()
    feature["ma50_to_ma200"] = ma50 / ma200 - 1.0

    for days in (20, 60, 120):
        feature[f"realized_vol_{days}d"] = (
            returns.rolling(days, min_periods=days).std() * math.sqrt(252)
        )

    for days in (20, 60, 120, 252):
        rolling_high = price.rolling(days, min_periods=days).max()
        feature[f"drawdown_{days}d"] = price / rolling_high - 1.0

    feature["rsi14"] = rsi(price, 14) / 100.0
    feature["vix_term_spread"] = (
        pd.to_numeric(history.get("VIX3M"), errors="coerce")
        - pd.to_numeric(history.get("VIX"), errors="coerce")
    )
    feature["vix_term_ratio"] = (
        pd.to_numeric(history.get("VIX"), errors="coerce")
        / pd.to_numeric(history.get("VIX3M"), errors="coerce").replace(0, np.nan)
    )

    # Acceleration and relative-strength style features.
    feature["momentum_acceleration_5_20"] = (
        feature["momentum_5d"] - feature["momentum_20d"] / 4.0
    )
    feature["momentum_acceleration_20_60"] = (
        feature["momentum_20d"] - feature["momentum_60d"] / 3.0
    )
    feature["vol_acceleration_20_60"] = (
        feature["realized_vol_20d"] - feature["realized_vol_60d"]
    )
    feature["drawdown_recovery_5d"] = (
        feature["momentum_5d"] - feature["drawdown_60d"].diff(5)
    )

    change_columns = [
        "Market_Regime",
        "Liquidity",
        "Positioning",
        "Buy_Score",
        "Risk_Score",
        "Margin_Score",
        "Breadth_Score",
        "NFL_13W_Change",
        "HY_OAS_Percentile",
        "VIX",
        "Net_Fed_Liquidity",
    ]
    for column in change_columns:
        if column in feature.columns:
            feature[f"{column}_change_5d"] = feature[column].diff(5)
            feature[f"{column}_change_20d"] = feature[column].diff(20)
            feature[f"{column}_acceleration"] = (
                feature[f"{column}_change_5d"]
                - feature[f"{column}_change_20d"] / 4.0
            )

    # Interactions intended for timing, not for the long-run risk cap.
    feature["trend_strength"] = (
        feature["price_to_ma200"].fillna(0.0)
        + feature["ma50_to_ma200"].fillna(0.0)
    )
    feature["credit_liquidity_interaction"] = (
        (100.0 - feature.get("HY_OAS_Percentile", pd.Series(50.0, index=feature.index)))
        * feature.get("Liquidity", pd.Series(50.0, index=feature.index))
        / 10_000.0
    )
    feature["breadth_trend_interaction"] = (
        feature.get("Breadth_Score", pd.Series(50.0, index=feature.index))
        * (feature["price_to_ma200"] > 0).astype(float)
        / 100.0
    )

    feature["expert"] = classify_expert(feature)
    feature["regime_key"] = classify_regime_key(feature)
    return feature.replace([np.inf, -np.inf], np.nan)


def classify_expert(feature: pd.DataFrame) -> pd.Series:
    bull = (
        feature["price_to_ma200"].fillna(-1.0) > 0
    ) & (
        feature["ma50_to_ma200"].fillna(-1.0) > 0
    )
    drawdown = feature["drawdown_60d"].fillna(0.0)
    pullback = bull & drawdown.between(-0.12, -0.02)
    expert = pd.Series("risk_off", index=feature.index, dtype=object)
    expert.loc[bull & ~pullback] = "trend"
    expert.loc[pullback] = "pullback"
    return expert


def classify_regime_key(feature: pd.DataFrame) -> pd.Series:
    trend = pd.Series("bear", index=feature.index, dtype=object)
    trend.loc[feature["price_to_ma200"].fillna(-1.0) > 0] = "neutral"
    trend.loc[
        (feature["price_to_ma200"].fillna(-1.0) > 0)
        & (feature["ma50_to_ma200"].fillna(-1.0) > 0)
    ] = "bull"

    vol = pd.Series("midvol", index=feature.index, dtype=object)
    vol.loc[feature["realized_vol_60d"].fillna(0.25) < 0.18] = "lowvol"
    vol.loc[feature["realized_vol_60d"].fillna(0.25) > 0.30] = "highvol"

    credit = pd.Series("midcredit", index=feature.index, dtype=object)
    credit_pct = feature.get(
        "HY_OAS_Percentile",
        pd.Series(50.0, index=feature.index),
    ).fillna(50.0)
    credit.loc[credit_pct < 35] = "easycredit"
    credit.loc[credit_pct > 70] = "stresscredit"

    pullback = pd.Series("shallow", index=feature.index, dtype=object)
    dd = feature["drawdown_60d"].fillna(0.0)
    pullback.loc[dd < -0.03] = "pullback"
    pullback.loc[dd < -0.09] = "deep"

    return trend + "|" + vol + "|" + credit + "|" + pullback

def future_net_returns(
    prices: pd.Series,
    cfg: ReturnAlphaConfig,
) -> Dict[int, pd.Series]:
    targets: Dict[int, pd.Series] = {}
    cost = cfg.round_trip_cost_bps / 10_000.0
    for horizon in cfg.horizons:
        gross = prices.shift(-horizon) / prices - 1.0
        borrow = cfg.annual_borrow_rate * horizon / 252.0
        targets[horizon] = gross - borrow - cost
    return targets


def point_in_time_static_baseline(
    net_return: pd.Series,
    horizon: int,
    cfg: ReturnAlphaConfig,
) -> pd.Series:
    """Rolling static-exposure return known at time t; no future labels."""
    known_value = net_return.shift(horizon + 1)
    values: deque = deque(maxlen=cfg.static_baseline_window_labels)
    result = pd.Series(np.nan, index=net_return.index, dtype=float)
    for date in net_return.index:
        value = known_value.get(date)
        if pd.notna(value):
            values.append(float(value))
        if len(values) >= cfg.static_baseline_min_samples:
            result.loc[date] = float(np.mean(values))
    return result


def build_relative_targets(
    prices: pd.Series,
    cfg: ReturnAlphaConfig,
) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
    """Target is borrowing return minus PIT static-exposure baseline."""
    net_targets = future_net_returns(prices, cfg)
    baseline: Dict[int, pd.Series] = {}
    relative: Dict[int, pd.Series] = {}
    for horizon, target in net_targets.items():
        baseline[horizon] = point_in_time_static_baseline(
            target,
            horizon,
            cfg,
        )
        relative[horizon] = target - baseline[horizon]
    return net_targets, baseline, relative


def weighted_standardize(
    frame: pd.DataFrame,
    weights: np.ndarray,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    values = frame.to_numpy(float)
    weights = weights / max(weights.sum(), 1e-12)
    means = pd.Series(np.sum(values * weights[:, None], axis=0), index=frame.columns)
    centered = values - means.to_numpy(float)
    variances = np.sum((centered ** 2) * weights[:, None], axis=0)
    stds = pd.Series(np.sqrt(np.maximum(variances, 1e-12)), index=frame.columns)
    stds = stds.replace(0, 1.0).fillna(1.0)
    return ((frame - means) / stds).clip(-8, 8), means, stds


def fit_ridge_forecaster(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_predict: pd.DataFrame,
    cfg: ReturnAlphaConfig,
) -> Tuple[pd.Series, pd.Series, pd.Series, Dict[str, Any]]:
    valid = y_train.notna()
    x_train = x_train.loc[valid]
    y_train = y_train.loc[valid].astype(float)
    if len(y_train) < 250:
        raise ValueError("Fewer than 250 valid labeled observations.")

    # Use lower-overlap samples while retaining the newest observation.
    stride = max(int(cfg.training_stride_days), 1)
    positions = np.arange(len(x_train))
    keep = (positions % stride == 0) | (positions == len(x_train) - 1)
    x_train = x_train.iloc[keep]
    y_train = y_train.iloc[keep]

    lower = float(y_train.quantile(cfg.target_winsor_lower))
    upper = float(y_train.quantile(cfg.target_winsor_upper))
    y_series = y_train.clip(lower, upper)

    medians = x_train.median(axis=0, skipna=True).fillna(0.0)
    train = x_train.fillna(medians).astype(float)
    predict = x_predict.fillna(medians).astype(float)

    ages = np.arange(len(train) - 1, -1, -1, dtype=float) * stride
    half_life = max(float(cfg.recency_half_life_days), 1.0)
    sample_weights = np.exp(-math.log(2.0) * ages / half_life)
    train_z, means, stds = weighted_standardize(train, sample_weights)
    predict_z = ((predict - means) / stds).clip(-8, 8)

    x = np.column_stack([np.ones(len(train_z)), train_z.to_numpy(float)])
    xp = np.column_stack([np.ones(len(predict_z)), predict_z.to_numpy(float)])
    y = y_series.to_numpy(float)
    sqrt_w = np.sqrt(sample_weights)
    xw = x * sqrt_w[:, None]
    yw = y * sqrt_w
    penalty = np.eye(x.shape[1]) * cfg.ridge_penalty
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(xw.T @ xw + penalty, xw.T @ yw)

    fitted = x @ beta
    residual = y - fitted
    residual_std = max(float(np.std(residual, ddof=1)), 1e-6)
    prediction = np.clip(xp @ beta, -0.30, 0.50)

    # Raw probability is empirical rather than a pure Gaussian extrapolation.
    empirical_probability = np.array([
        float(np.mean(residual > -value))
        for value in prediction
    ])
    gaussian_probability = norm.cdf(prediction / residual_std)
    raw_probability = 0.5 * empirical_probability + 0.5 * gaussian_probability

    target_values = y_series.to_numpy(float)
    percentile = np.array([
        float(np.mean(target_values <= value))
        for value in prediction
    ])

    diagnostics = {
        "train_rows_before_stride": int(valid.sum()),
        "train_rows_after_stride": int(len(y_series)),
        "residual_std": residual_std,
        "target_mean": float(np.mean(y)),
        "target_positive_rate": float(np.mean(y > 0)),
        "feature_count": int(train_z.shape[1]),
    }
    return (
        pd.Series(prediction, index=x_predict.index),
        pd.Series(raw_probability, index=x_predict.index).clip(0.01, 0.99),
        pd.Series(percentile, index=x_predict.index).clip(0.0, 1.0),
        diagnostics,
    )


def monthly_refit_dates(index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    frame = pd.DataFrame(index=index)
    return list(frame.groupby([index.year, index.month]).head(1).index)


def model_feature_columns(features: pd.DataFrame) -> List[str]:
    return [
        column
        for column in features.columns
        if column not in {"expert", "regime_key"}
    ]


def expert_train_index(
    features: pd.DataFrame,
    train_index: pd.DatetimeIndex,
    expert: str,
    cfg: ReturnAlphaConfig,
) -> pd.DatetimeIndex:
    if expert == "risk_off":
        return pd.DatetimeIndex([])
    same = train_index[features.loc[train_index, "expert"].astype(str) == expert]
    if len(same) >= cfg.expert_min_train_rows:
        return same
    bull = train_index[
        features.loc[train_index, "expert"].astype(str).isin(["trend", "pullback"])
    ]
    if len(bull) >= cfg.expert_min_train_rows:
        return bull
    return train_index


def generate_oos_forecasts(
    history: pd.DataFrame,
    prices: pd.Series,
    cfg: ReturnAlphaConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    features = build_features(history, prices)
    prices = prices.reindex(features.index)
    _, baselines, relative_targets = build_relative_targets(prices, cfg)
    columns = model_feature_columns(features)

    output = pd.DataFrame(index=features.index)
    model_records: List[Dict[str, Any]] = []
    refit_dates = monthly_refit_dates(features.index)

    for refit_index, prediction_start in enumerate(refit_dates):
        start_position = features.index.get_loc(prediction_start)
        if refit_index + 1 < len(refit_dates):
            prediction_end = refit_dates[refit_index + 1] - pd.Timedelta(days=1)
            prediction_index = features.loc[prediction_start:prediction_end].index
        else:
            prediction_index = features.loc[prediction_start:].index

        month_pred = pd.DataFrame(index=prediction_index)
        month_pred["expert"] = features.loc[prediction_index, "expert"]
        all_horizons_available = True
        horizon_train_ends: List[pd.Timestamp] = []

        for horizon, horizon_weight in zip(cfg.horizons, cfg.horizon_weights):
            label_end_position = start_position - horizon - 1
            if label_end_position < cfg.min_train_days - 1:
                all_horizons_available = False
                break
            train_start_position = max(
                0,
                label_end_position - cfg.train_window_days + 1,
            )
            full_train_index = features.index[
                train_start_position:label_end_position + 1
            ]
            horizon_train_ends.append(full_train_index[-1])

            pred_h = pd.Series(np.nan, index=prediction_index, dtype=float)
            rank_h = pd.Series(np.nan, index=prediction_index, dtype=float)
            prob_h = pd.Series(np.nan, index=prediction_index, dtype=float)

            for expert in ("trend", "pullback", "risk_off"):
                pred_index = prediction_index[
                    month_pred["expert"] == expert
                ]
                if len(pred_index) == 0:
                    continue
                if expert == "risk_off":
                    pred_h.loc[pred_index] = 0.0
                    rank_h.loc[pred_index] = 0.0
                    prob_h.loc[pred_index] = 0.01
                    continue

                train_index = expert_train_index(
                    features,
                    full_train_index,
                    expert,
                    cfg,
                )
                pred, probability, percentile, diagnostics = fit_ridge_forecaster(
                    features.loc[train_index, columns],
                    relative_targets[horizon].loc[train_index],
                    features.loc[pred_index, columns],
                    cfg,
                )
                pred_h.loc[pred_index] = pred * (60.0 / horizon)
                rank_h.loc[pred_index] = percentile
                prob_h.loc[pred_index] = probability
                model_records.append({
                    "prediction_start": prediction_start.date().isoformat(),
                    "prediction_end": prediction_index[-1].date().isoformat(),
                    "train_start": train_index[0].date().isoformat(),
                    "train_end": train_index[-1].date().isoformat(),
                    "purge_trading_days": horizon + 1,
                    "horizon_days": horizon,
                    "expert": expert,
                    "horizon_weight": horizon_weight,
                    **diagnostics,
                })

            month_pred[f"relative_alpha_{horizon}"] = pred_h
            month_pred[f"forecast_rank_{horizon}"] = rank_h
            month_pred[f"diagnostic_probability_{horizon}"] = prob_h
            month_pred[f"static_baseline_{horizon}"] = (
                baselines[horizon].reindex(prediction_index)
                * (60.0 / horizon)
            )

        if not all_horizons_available:
            continue

        weights = np.asarray(cfg.horizon_weights, dtype=float)
        weights = weights / weights.sum()
        relative_frame = month_pred[
            [f"relative_alpha_{h}" for h in cfg.horizons]
        ]
        rank_frame = month_pred[
            [f"forecast_rank_{h}" for h in cfg.horizons]
        ]
        probability_frame = month_pred[
            [f"diagnostic_probability_{h}" for h in cfg.horizons]
        ]
        baseline_frame = month_pred[
            [f"static_baseline_{h}" for h in cfg.horizons]
        ]

        combined_relative = relative_frame.mul(weights, axis=1).sum(axis=1)
        combined_rank = rank_frame.mul(weights, axis=1).sum(axis=1)
        combined_probability = probability_frame.mul(weights, axis=1).sum(axis=1)
        combined_baseline = baseline_frame.mul(weights, axis=1).sum(axis=1)

        output.loc[prediction_index, "expected_relative_alpha_60d"] = combined_relative
        output.loc[prediction_index, "static_baseline_60d"] = combined_baseline
        output.loc[prediction_index, "expected_net_return_60d"] = (
            combined_baseline + combined_relative
        )
        output.loc[prediction_index, "forecast_rank"] = combined_rank.clip(0.0, 1.0)
        output.loc[prediction_index, "diagnostic_probability"] = combined_probability.clip(0.01, 0.99)
        output.loc[prediction_index, "expert"] = month_pred["expert"]
        output.loc[prediction_index, "model_train_end"] = (
            max(horizon_train_ends).date().isoformat()
        )

    lookback = max(int(cfg.rank_stability_lookback_days), 1)
    minimum = min(
        max(int(cfg.rank_stability_min_observations), 1),
        lookback,
    )
    output["stable_forecast_rank"] = (
        output["forecast_rank"]
        .rolling(lookback, min_periods=minimum)
        .median()
    )
    output["stable_expected_relative_alpha_60d"] = (
        output["expected_relative_alpha_60d"]
        .rolling(lookback, min_periods=minimum)
        .mean()
    )
    output["rank_pass_count"] = (
        (output["forecast_rank"] >= cfg.rank_tier_1)
        .astype(float)
        .rolling(lookback, min_periods=minimum)
        .sum()
    )

    passthrough = [
        "price_to_ma200",
        "ma50_to_ma200",
        "drawdown_60d",
        "momentum_20d",
        "Breadth_Score_change_5d",
        "Risk_Score_change_5d",
    ]
    for column in passthrough:
        if column in features:
            output[column] = features[column]
    return output, pd.DataFrame(model_records)


def generate_latest_forecast(
    history: pd.DataFrame,
    prices: pd.Series,
    cfg: ReturnAlphaConfig,
) -> Tuple[pd.Series, pd.DataFrame]:
    forecasts, records = generate_oos_forecasts(history, prices, cfg)
    latest_date = history.index.intersection(forecasts.index)[-1]
    row = forecasts.loc[latest_date]
    if pd.isna(row.get("expected_relative_alpha_60d")):
        raise RuntimeError("Latest Return Alpha forecast is unavailable.")
    return row, records

def risk_cap_from_scores(row: pd.Series, cfg: ReturnAlphaConfig) -> Tuple[float, List[str]]:
    risk = float(row.get("Risk_Score")) if pd.notna(row.get("Risk_Score")) else 100.0
    liquidity = float(row.get("Liquidity")) if pd.notna(row.get("Liquidity")) else 0.0
    margin_score = float(row.get("Margin_Score")) if pd.notna(row.get("Margin_Score")) else 0.0
    coverage = float(row.get("Coverage_Ratio")) if pd.notna(row.get("Coverage_Ratio")) else 0.0
    regime = float(row.get("Market_Regime")) if pd.notna(row.get("Market_Regime")) else 0.0
    backwardation_value = row.get("VIX_Backwardation", 0.0)
    backwardation = (
        pd.notna(backwardation_value)
        and float(backwardation_value) > 0.5
    )
    reasons: List[str] = []

    if risk > 70 or liquidity < 30 or margin_score < 45 or coverage < 0.60:
        return 0.0, ["Hard_Risk_Cap_Zero"]

    if margin_score < 50:
        cap = cfg.risk_cap_margin_45_50
    elif margin_score < 55:
        cap = cfg.risk_cap_margin_50_55
    elif margin_score < 60:
        cap = cfg.risk_cap_margin_55_60
    elif margin_score < 70:
        cap = cfg.risk_cap_margin_60_70
    else:
        cap = cfg.risk_cap_margin_70_plus

    if risk > 55:
        cap -= cfg.risk_penalty_if_risk_above_55
        reasons.append("Risk_Above_55_Penalty")
    if liquidity < 45:
        cap -= cfg.risk_penalty_if_liquidity_below_45
        reasons.append("Liquidity_Below_45_Penalty")
    if coverage < 0.80:
        cap = min(cap, cfg.coverage_cap_if_below_80)
        reasons.append("Coverage_Below_80_Cap")
    if regime < 45:
        cap = min(cap, cfg.regime_cap_if_below_45)
        reasons.append("Weak_Regime_Cap")
    if backwardation:
        cap = min(cap, cfg.backwardation_cap_pct)
        reasons.append("VIX_Backwardation_Cap")

    return max(0.0, min(cap, cfg.max_alpha_margin_pct)), reasons


def alpha_demand_from_forecast(
    forecast_row: pd.Series,
    cfg: ReturnAlphaConfig,
) -> Tuple[float, List[str]]:
    relative_alpha = forecast_row.get("stable_expected_relative_alpha_60d")
    expected_net = forecast_row.get("expected_net_return_60d")
    forecast_rank = forecast_row.get("forecast_rank")
    stable_rank = forecast_row.get("stable_forecast_rank")
    pass_count = forecast_row.get("rank_pass_count")
    expert = str(forecast_row.get("expert", "risk_off"))

    required = (
        relative_alpha,
        expected_net,
        forecast_rank,
        stable_rank,
        pass_count,
    )
    if any(pd.isna(value) for value in required):
        return 0.0, ["Forecast_Unavailable"]
    if expert == "risk_off":
        return cfg.risk_off_margin_pct, ["Risk_Off_Expert"]

    relative_pct = 100.0 * float(relative_alpha)
    net_pct = 100.0 * float(expected_net)
    stable_rank = float(stable_rank)
    pass_count = int(round(float(pass_count)))
    demand = 0.0
    tier_name = "No_Rank_Tier"

    tiers = [
        (
            cfg.rank_tier_1,
            cfg.relative_alpha_tier_1_pct,
            cfg.margin_tier_1_pct,
            "Tier_1",
        ),
        (
            cfg.rank_tier_2,
            cfg.relative_alpha_tier_2_pct,
            cfg.margin_tier_2_pct,
            "Tier_2",
        ),
        (
            cfg.rank_tier_3,
            cfg.relative_alpha_tier_3_pct,
            cfg.margin_tier_3_pct,
            "Tier_3",
        ),
        (
            cfg.rank_tier_4,
            cfg.relative_alpha_tier_4_pct,
            cfg.margin_tier_4_pct,
            "Tier_4",
        ),
    ]
    for rank_threshold, alpha_threshold, margin, name in tiers:
        if (
            stable_rank >= rank_threshold
            and relative_pct >= alpha_threshold
            and net_pct > 0
            and pass_count >= cfg.rank_stability_min_passes
        ):
            demand = margin
            tier_name = name

    reasons = [
        tier_name,
        f"Expert_{expert}",
        f"Stable_Rank_{stable_rank:.3f}",
        f"Rank_Passes_{pass_count}",
    ]
    above_ma200 = float(forecast_row.get("price_to_ma200", -1.0)) > 0
    ma_trend = float(forecast_row.get("ma50_to_ma200", -1.0)) > 0
    if cfg.require_price_above_ma200 and not (above_ma200 and ma_trend):
        return 0.0, reasons + ["Long_Trend_Filter_Failed"]
    return min(demand, cfg.max_alpha_margin_pct), reasons


def build_alpha_targets(
    history: pd.DataFrame,
    forecasts: pd.DataFrame,
    cfg: ReturnAlphaConfig,
) -> Tuple[pd.Series, pd.DataFrame]:
    raw_targets: Dict[pd.Timestamp, float] = {}
    records: List[Dict[str, Any]] = []
    forecast_dates = history.index.intersection(forecasts.index)
    valid_dates = [
        date
        for date in forecast_dates
        if pd.notna(forecasts.loc[date].get("expected_relative_alpha_60d"))
    ]
    valid_position = {date: position for position, date in enumerate(valid_dates)}
    interval = max(int(cfg.decision_interval_business_days), 1)
    carried_demand = 0.0
    last_model_decision_date: Optional[str] = None

    for date in forecast_dates:
        forecast_row = forecasts.loc[date]
        history_row = history.loc[date]
        position = valid_position.get(date)
        decision_due = position is not None and position % interval == 0

        if decision_due:
            carried_demand, alpha_reasons = alpha_demand_from_forecast(
                forecast_row,
                cfg,
            )
            last_model_decision_date = date.date().isoformat()
        else:
            alpha_reasons = ["Carry_Previous_Scheduled_Demand"]

        risk_cap, risk_reasons = risk_cap_from_scores(history_row, cfg)
        target = min(carried_demand, risk_cap)
        raw_targets[date] = target
        records.append({
            "date": date.date().isoformat(),
            "model_version": cfg.model_version,
            "decision_due": bool(decision_due),
            "last_model_decision_date": last_model_decision_date,
            "decision_interval_business_days": interval,
            "expert": forecast_row.get("expert"),
            "expected_relative_alpha_60d_pct": (
                100.0 * float(forecast_row["expected_relative_alpha_60d"])
                if pd.notna(forecast_row.get("expected_relative_alpha_60d"))
                else None
            ),
            "stable_expected_relative_alpha_60d_pct": (
                100.0 * float(forecast_row["stable_expected_relative_alpha_60d"])
                if pd.notna(forecast_row.get("stable_expected_relative_alpha_60d"))
                else None
            ),
            "static_baseline_60d_pct": (
                100.0 * float(forecast_row["static_baseline_60d"])
                if pd.notna(forecast_row.get("static_baseline_60d"))
                else None
            ),
            "expected_net_return_60d_pct": (
                100.0 * float(forecast_row["expected_net_return_60d"])
                if pd.notna(forecast_row.get("expected_net_return_60d"))
                else None
            ),
            "forecast_rank": (
                float(forecast_row["forecast_rank"])
                if pd.notna(forecast_row.get("forecast_rank"))
                else None
            ),
            "stable_forecast_rank": (
                float(forecast_row["stable_forecast_rank"])
                if pd.notna(forecast_row.get("stable_forecast_rank"))
                else None
            ),
            "rank_pass_count": (
                int(round(float(forecast_row["rank_pass_count"])))
                if pd.notna(forecast_row.get("rank_pass_count"))
                else None
            ),
            "diagnostic_probability": (
                float(forecast_row["diagnostic_probability"])
                if pd.notna(forecast_row.get("diagnostic_probability"))
                else None
            ),
            "alpha_demand_margin_pct": carried_demand,
            "risk_cap_margin_pct": risk_cap,
            "alpha_raw_target_margin_pct": target,
            "alpha_reasons": alpha_reasons,
            "risk_reasons": risk_reasons,
            "model_train_end": forecast_row.get("model_train_end"),
        })
    return (
        pd.Series(
            raw_targets,
            name="Alpha_Raw_Target_Margin_Pct",
        ).sort_index(),
        pd.DataFrame(records),
    )


def business_days_between_dates(start: Optional[str], end: pd.Timestamp) -> int:
    if not start:
        return 10 ** 9
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    if end_date <= start_date:
        return 0
    return int(np.busday_count(start_date, end_date))


def alpha_hard_exit_reasons(
    date: pd.Timestamp,
    price: float,
    forecast: pd.Series,
    row: pd.Series,
    state: ReturnAlphaState,
    cfg: ReturnAlphaConfig,
    state_cfg: StateMachineConfig,
) -> List[str]:
    reasons: List[str] = []
    risk = float(row.get("Risk_Score")) if pd.notna(row.get("Risk_Score")) else 100.0
    liquidity = float(row.get("Liquidity")) if pd.notna(row.get("Liquidity")) else 0.0
    margin_score = float(row.get("Margin_Score")) if pd.notna(row.get("Margin_Score")) else 0.0
    coverage = float(row.get("Coverage_Ratio")) if pd.notna(row.get("Coverage_Ratio")) else 0.0
    stable_rank = float(forecast.get("stable_forecast_rank")) if pd.notna(forecast.get("stable_forecast_rank")) else 0.0
    relative_pct = 100.0 * float(forecast.get("stable_expected_relative_alpha_60d")) if pd.notna(forecast.get("stable_expected_relative_alpha_60d")) else -999.0
    expert = str(forecast.get("expert", "risk_off"))

    if risk > state_cfg.hard_exit_risk_score:
        reasons.append("Risk_Score_Hard_Exit")
    if liquidity < state_cfg.hard_exit_liquidity_score:
        reasons.append("Liquidity_Hard_Exit")
    if margin_score < state_cfg.hard_exit_margin_score:
        reasons.append("Margin_Score_Hard_Exit")
    if coverage < state_cfg.hard_exit_coverage_ratio:
        reasons.append("Coverage_Hard_Exit")

    if state.model_actual_margin_pct > 0:
        if stable_rank < cfg.exit_rank_percentile:
            reasons.append("Rank_Exit")
        if relative_pct <= cfg.exit_relative_alpha_pct:
            reasons.append("Relative_Alpha_Exit")
        if expert == "risk_off":
            reasons.append("Risk_Off_Expert_Exit")
        if state.entry_date and state.entry_price:
            held = business_days_between_dates(state.entry_date, date)
            return_pct = 100.0 * (price / float(state.entry_price) - 1.0)
            if (
                held >= cfg.time_stop_business_days
                and return_pct <= cfg.time_stop_min_return_pct
            ):
                reasons.append("Time_Stop_No_Follow_Through")
    return reasons


def update_alpha_state(
    date: pd.Timestamp,
    price: float,
    raw_target: float,
    forecast: pd.Series,
    row: pd.Series,
    state: ReturnAlphaState,
    cfg: ReturnAlphaConfig,
    state_cfg: StateMachineConfig,
    decision_due: bool = False,
    scheduled_demand: Optional[float] = None,
) -> Tuple[ReturnAlphaState, Dict[str, Any]]:
    next_state = copy.deepcopy(state)
    signal_date = pd.Timestamp(date).normalize()
    signal_date_str = signal_date.date().isoformat()
    current = float(next_state.model_actual_margin_pct)
    raw_target = float(np.clip(raw_target, 0.0, state_cfg.max_target_margin_pct))

    if next_state.last_processed_signal_date == signal_date_str:
        return next_state, {
            "signal_date": signal_date_str,
            "raw_target_margin_pct": raw_target,
            "confirmed_target_margin_pct": current,
            "model_actual_margin_pct": current,
            "action": "NO_NEW_SIGNAL",
            "reasons": ["Signal_Date_Already_Processed"],
            "hard_exit": False,
            "decision_due": bool(decision_due),
            "last_model_decision_date": next_state.last_model_decision_date,
        }

    if decision_due:
        next_state.last_model_decision_date = signal_date_str
        if scheduled_demand is not None:
            next_state.scheduled_alpha_demand_margin_pct = float(scheduled_demand)
        if pd.notna(forecast.get("stable_forecast_rank")):
            next_state.scheduled_forecast_rank = float(
                forecast.get("stable_forecast_rank")
            )
        if pd.notna(forecast.get("stable_expected_relative_alpha_60d")):
            next_state.scheduled_expected_relative_alpha_pct = 100.0 * float(
                forecast.get("stable_expected_relative_alpha_60d")
            )

    if current > 0:
        next_state.peak_price = max(float(next_state.peak_price or price), price)

    hard_reasons = alpha_hard_exit_reasons(
        signal_date,
        price,
        forecast,
        row,
        next_state,
        cfg,
        state_cfg,
    )
    action = "HOLD"
    reasons: List[str] = []
    hard_exit = bool(hard_reasons)

    if hard_exit:
        next_state.model_actual_margin_pct = 0.0
        next_state.confirmed_target_margin_pct = 0.0
        next_state.increase_streak = 0
        next_state.decrease_streak = 0
        next_state.entry_date = None
        next_state.entry_price = None
        next_state.peak_price = None
        action = "EXIT_ALL" if current > 0 else "HOLD_ZERO"
        reasons = hard_reasons
        if current > 0:
            next_state.last_trade_date = signal_date_str
    else:
        breadth_change = forecast.get("Breadth_Score_change_5d")
        risk_change = forecast.get("Risk_Score_change_5d")
        shock = (
            current > 0
            and (
                (pd.notna(breadth_change) and float(breadth_change) <= cfg.breadth_shock_5d_threshold)
                or (pd.notna(risk_change) and float(risk_change) >= cfg.risk_score_shock_5d_threshold)
            )
        )
        if shock:
            new_target = max(0.0, min(raw_target, current * cfg.shock_reduce_fraction))
            next_state.model_actual_margin_pct = new_target
            next_state.confirmed_target_margin_pct = new_target
            next_state.increase_streak = 0
            next_state.decrease_streak = 0
            next_state.last_trade_date = signal_date_str
            action = "SHOCK_REDUCE"
            reasons = ["Breadth_Or_Risk_Shock"]
        elif raw_target <= 0 and current > 0:
            next_state.model_actual_margin_pct = 0.0
            next_state.confirmed_target_margin_pct = 0.0
            next_state.increase_streak = 0
            next_state.decrease_streak = 0
            next_state.last_trade_date = signal_date_str
            next_state.entry_date = None
            next_state.entry_price = None
            next_state.peak_price = None
            action = "REPAY_ALL"
            reasons = ["Alpha_Demand_Zero"]
        else:
            difference = raw_target - current
            if abs(difference) < state_cfg.no_trade_band_pct:
                next_state.increase_streak = 0
                next_state.decrease_streak = 0
                action = "HOLD"
                reasons = ["Inside_No_Trade_Band"]
            elif difference >= state_cfg.increase_trigger_pct:
                next_state.increase_streak += 1
                next_state.decrease_streak = 0
                days_since_trade = business_days_between_dates(next_state.last_trade_date, signal_date)
                days_since_increase = business_days_between_dates(next_state.last_increase_date, signal_date)
                week_key = f"{signal_date.isocalendar().year}-W{signal_date.isocalendar().week:02d}"
                weekly_ok = (
                    not state_cfg.increase_once_per_iso_week
                    or next_state.last_increase_iso_week != week_key
                )
                eligible = (
                    next_state.increase_streak >= state_cfg.increase_confirm_days
                    and days_since_trade >= state_cfg.min_business_days_between_trades
                    and days_since_increase >= state_cfg.increase_cooldown_business_days
                    and weekly_ok
                )
                if eligible:
                    new_target = min(
                        raw_target,
                        current + state_cfg.max_increase_step_pct,
                        state_cfg.max_target_margin_pct,
                    )
                    next_state.model_actual_margin_pct = new_target
                    next_state.confirmed_target_margin_pct = new_target
                    next_state.last_trade_date = signal_date_str
                    next_state.last_increase_date = signal_date_str
                    next_state.last_increase_iso_week = week_key
                    next_state.increase_streak = 0
                    if current <= 0 and new_target > 0:
                        next_state.entry_date = signal_date_str
                        next_state.entry_price = float(price)
                        next_state.peak_price = float(price)
                    action = "INCREASE_MARGIN"
                    reasons = ["Increase_Executed"]
                else:
                    action = "WAIT_INCREASE_CONFIRMATION"
                    reasons = [
                        f"Increase_Confirmation_{next_state.increase_streak}_of_{state_cfg.increase_confirm_days}"
                    ]
            elif difference <= -state_cfg.decrease_trigger_pct:
                next_state.decrease_streak += 1
                next_state.increase_streak = 0
                if next_state.decrease_streak >= state_cfg.decrease_confirm_days:
                    new_target = max(
                        raw_target,
                        current - state_cfg.max_decrease_step_pct,
                        0.0,
                    )
                    next_state.model_actual_margin_pct = new_target
                    next_state.confirmed_target_margin_pct = new_target
                    next_state.last_trade_date = signal_date_str
                    next_state.decrease_streak = 0
                    action = "DECREASE_MARGIN" if new_target > 0 else "REPAY_ALL"
                    reasons = ["Decrease_Executed"]
                    if new_target <= 0:
                        next_state.entry_date = None
                        next_state.entry_price = None
                        next_state.peak_price = None
                else:
                    action = "WAIT_DECREASE_CONFIRMATION"
                    reasons = [
                        f"Decrease_Confirmation_{next_state.decrease_streak}_of_{state_cfg.decrease_confirm_days}"
                    ]
            else:
                action = "HOLD"
                reasons = ["No_Trigger"]

    next_state.last_processed_signal_date = signal_date_str
    next_state.last_action = action
    next_state.last_reason = reasons
    decision = {
        "signal_date": signal_date_str,
        "raw_target_margin_pct": round(raw_target, 4),
        "confirmed_target_margin_pct": round(next_state.confirmed_target_margin_pct, 4),
        "model_actual_margin_pct": round(next_state.model_actual_margin_pct, 4),
        "action": action,
        "reasons": reasons,
        "hard_exit": hard_exit,
        "increase_streak": next_state.increase_streak,
        "decrease_streak": next_state.decrease_streak,
        "entry_date": next_state.entry_date,
        "entry_price": next_state.entry_price,
        "peak_price": next_state.peak_price,
        "decision_due": bool(decision_due),
        "last_model_decision_date": next_state.last_model_decision_date,
        "scheduled_alpha_demand_margin_pct": next_state.scheduled_alpha_demand_margin_pct,
    }
    return next_state, decision


def apply_state_machine(
    history: pd.DataFrame,
    prices: pd.Series,
    forecasts: pd.DataFrame,
    raw_target: pd.Series,
    decision_records: pd.DataFrame,
    cfg: ReturnAlphaConfig,
    state_cfg: StateMachineConfig,
) -> Tuple[pd.Series, pd.DataFrame]:
    state = ReturnAlphaState()
    targets: Dict[pd.Timestamp, float] = {}
    records: List[Dict[str, Any]] = []
    metadata = pd.DataFrame()
    if not decision_records.empty:
        metadata = decision_records.copy()
        metadata["date"] = pd.to_datetime(metadata["date"], errors="coerce")
        metadata = metadata.dropna(subset=["date"]).set_index("date")

    for date in history.index:
        row = history.loc[date]
        forecast = forecasts.loc[date] if date in forecasts.index else pd.Series(dtype=float)
        decision_due = bool(
            metadata.loc[date, "decision_due"]
            if date in metadata.index
            else False
        )
        scheduled_demand = (
            float(metadata.loc[date, "alpha_demand_margin_pct"])
            if date in metadata.index
            else None
        )
        state, decision = update_alpha_state(
            date,
            float(prices.loc[date]),
            float(raw_target.get(date, 0.0)),
            forecast,
            row,
            state,
            cfg,
            state_cfg,
            decision_due=decision_due,
            scheduled_demand=scheduled_demand,
        )
        targets[date] = state.model_actual_margin_pct
        records.append({"date": date.date().isoformat(), **decision})
    return (
        pd.Series(targets, name="Alpha_Dynamic_Target_Margin_Pct"),
        pd.DataFrame(records),
    )


def compare_strategies(
    history: pd.DataFrame,
    prices: pd.Series,
    alpha_raw: pd.Series,
    alpha_dynamic: pd.Series,
    current_state_cfg: StateMachineConfig,
    bt_cfg: BacktestConfig,
    evaluation_start: pd.Timestamp,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Compare every strategy over the same genuine OOS date range."""

    full_history = history.copy()
    full_current_dynamic, _ = generate_dynamic_targets(
        full_history,
        current_state_cfg,
    )

    history = full_history.loc[evaluation_start:].copy()
    prices = prices.reindex(history.index).dropna()
    history = history.reindex(prices.index)

    targets: Dict[str, pd.Series] = {
        "no_margin": pd.Series(0.0, index=history.index),
        "fixed_1pct": pd.Series(1.0, index=history.index),
        "fixed_2pct": pd.Series(2.0, index=history.index),
        "fixed_3pct": pd.Series(3.0, index=history.index),
        "fixed_5pct": pd.Series(5.0, index=history.index),
        "current_raw_target": history["Target_Margin_Pct"].fillna(0.0),
        "current_dynamic_state": full_current_dynamic.reindex(history.index).fillna(0.0),
        "return_alpha_raw": alpha_raw.reindex(history.index).fillna(0.0),
        "return_alpha_dynamic": alpha_dynamic.reindex(history.index).fillna(0.0),
    }

    ledgers: Dict[str, pd.DataFrame] = {}
    for name, target in targets.items():
        ledgers[name] = simulate_account(prices, target, bt_cfg, name)

    average_exposure = float(
        ledgers["return_alpha_dynamic"][
            "effective_target_margin_pct"
        ].mean()
    )
    matched_target = pd.Series(average_exposure, index=history.index)
    ledgers["fixed_exposure_matched"] = simulate_account(
        prices,
        matched_target,
        bt_cfg,
        "fixed_exposure_matched",
    )

    metric_rows = [
        account_metrics(ledger, bt_cfg.trading_days_per_year)
        for ledger in ledgers.values()
    ]
    return ledgers, pd.DataFrame(metric_rows)


def newey_west_t_stat(series: pd.Series, max_lag: int = 10) -> Dict[str, Optional[float]]:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    n = len(x)
    if n < 50:
        return {"mean_daily": None, "annualized_mean": None, "t_stat": None}
    centered = x - x.mean()
    gamma0 = float(centered @ centered / n)
    variance = gamma0
    for lag in range(1, min(max_lag, n - 1) + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(centered[lag:] @ centered[:-lag] / n)
        variance += 2.0 * weight * gamma
    standard_error = math.sqrt(max(variance, 0.0) / n)
    t_stat = float(x.mean() / standard_error) if standard_error > 0 else None
    return {
        "mean_daily": float(x.mean()),
        "annualized_mean": float(x.mean() * 252.0),
        "t_stat": t_stat,
    }


def block_bootstrap_monthly_difference(
    strategy: pd.Series,
    benchmark: pd.Series,
    iterations: int,
    block_months: int,
    seed: int,
) -> Dict[str, Optional[float]]:
    aligned = pd.concat([strategy, benchmark], axis=1, join="inner").dropna()
    if aligned.empty:
        return {"annualized_mean_diff": None, "ci_low": None, "ci_high": None, "probability_positive": None}
    monthly = (1.0 + aligned).resample("ME").prod() - 1.0
    diff = (monthly.iloc[:, 0] - monthly.iloc[:, 1]).dropna().to_numpy(float)
    n = len(diff)
    if n < 24:
        return {"annualized_mean_diff": None, "ci_low": None, "ci_high": None, "probability_positive": None}
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations)
    for iteration in range(iterations):
        sample: List[float] = []
        while len(sample) < n:
            start = int(rng.integers(0, n))
            for offset in range(block_months):
                sample.append(diff[(start + offset) % n])
                if len(sample) >= n:
                    break
        estimates[iteration] = np.mean(sample[:n]) * 12.0
    return {
        "annualized_mean_diff": float(np.mean(diff) * 12.0),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "probability_positive": float(np.mean(estimates > 0)),
    }


def forecast_diagnostics(
    prices: pd.Series,
    alpha_target: pd.Series,
    horizon: int = 60,
) -> Dict[str, Any]:
    future = prices.shift(-horizon) / prices - 1.0
    target = alpha_target.reindex(prices.index).fillna(0.0)
    active = target > 0
    inactive = ~active
    return {
        "active_days": int(active.sum()),
        "active_fraction": float(active.mean()),
        "active_future_return_mean": float(future[active].mean()),
        "inactive_future_return_mean": float(future[inactive].mean()),
        "active_positive_rate": float((future[active] > 0).mean()),
        "inactive_positive_rate": float((future[inactive] > 0).mean()),
        "active_minus_inactive_future_return": float(future[active].mean() - future[inactive].mean()),
    }


def rank_forecast_diagnostics(
    forecasts: pd.DataFrame,
    prices: pd.Series,
    decision_records: pd.DataFrame,
    cfg: ReturnAlphaConfig,
    horizon: int = 60,
) -> Dict[str, Any]:
    future_net = (
        prices.shift(-horizon) / prices - 1.0
        - cfg.annual_borrow_rate * horizon / 252.0
        - cfg.round_trip_cost_bps / 10_000.0
    )
    realized_relative = future_net - forecasts.get(
        "static_baseline_60d",
        pd.Series(np.nan, index=forecasts.index),
    )
    decision_dates = pd.DatetimeIndex([])
    if not decision_records.empty:
        records = decision_records.copy()
        records["date"] = pd.to_datetime(records["date"], errors="coerce")
        decision_dates = pd.DatetimeIndex(
            records.loc[records["decision_due"].astype(bool), "date"].dropna()
        )
    aligned = pd.concat(
        [
            forecasts.get("stable_forecast_rank").rename("rank"),
            forecasts.get("stable_expected_relative_alpha_60d").rename("prediction"),
            realized_relative.rename("realized"),
        ],
        axis=1,
    ).dropna()
    if len(decision_dates):
        aligned = aligned.loc[aligned.index.intersection(decision_dates)]
    if len(aligned) < 20:
        return {
            "samples": int(len(aligned)),
            "spearman_rank_ic": None,
            "top_decile_realized_relative_alpha": None,
            "rest_realized_relative_alpha": None,
            "top_decile_positive_rate": None,
            "rank_bucket_table": [],
        }

    rank_ic = spearmanr(
        aligned["rank"].to_numpy(float),
        aligned["realized"].to_numpy(float),
        nan_policy="omit",
    ).statistic
    cutoff = float(aligned["rank"].quantile(0.90))
    top = aligned[aligned["rank"] >= cutoff]
    rest = aligned[aligned["rank"] < cutoff]
    bucket = pd.qcut(
        aligned["rank"],
        q=min(5, aligned["rank"].nunique()),
        duplicates="drop",
    )
    table_frame = aligned.assign(bucket=bucket)
    bucket_rows = [
        {
            "bucket": str(name),
            "count": int(len(group)),
            "mean_rank": float(group["rank"].mean()),
            "mean_predicted_relative_alpha": float(group["prediction"].mean()),
            "mean_realized_relative_alpha": float(group["realized"].mean()),
            "positive_realized_rate": float((group["realized"] > 0).mean()),
        }
        for name, group in table_frame.groupby("bucket", observed=True)
    ]
    realized_means = [row["mean_realized_relative_alpha"] for row in bucket_rows]
    monotonic_steps = sum(
        right >= left
        for left, right in zip(realized_means, realized_means[1:])
    )
    return {
        "samples": int(len(aligned)),
        "spearman_rank_ic": (
            float(rank_ic) if pd.notna(rank_ic) else None
        ),
        "top_decile_cutoff": cutoff,
        "top_decile_count": int(len(top)),
        "top_decile_realized_relative_alpha": float(top["realized"].mean()),
        "rest_realized_relative_alpha": float(rest["realized"].mean()),
        "top_decile_positive_rate": float((top["realized"] > 0).mean()),
        "monotonic_bucket_steps": int(monotonic_steps),
        "possible_monotonic_steps": max(len(bucket_rows) - 1, 0),
        "rank_bucket_table": bucket_rows,
    }


def fold_analysis(ledgers: Mapping[str, pd.DataFrame], fold_years: int) -> pd.DataFrame:
    alpha = ledgers["return_alpha_dynamic"]
    start_year = alpha.index.min().year
    end_year = alpha.index.max().year
    rows: List[Dict[str, Any]] = []
    for year in range(start_year, end_year + 1, fold_years):
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{min(year + fold_years - 1, end_year)}-12-31")
        row: Dict[str, Any] = {"fold_start": start.date().isoformat(), "fold_end": end.date().isoformat()}
        for name in ("no_margin", "fixed_exposure_matched", "return_alpha_dynamic"):
            ledger = ledgers[name].loc[start:end]
            if len(ledger) < 50:
                row[f"{name}_cagr"] = None
                continue
            normalized = ledger.copy()
            normalized["equity"] = 1_000_000.0 * (1.0 + normalized["strategy_return"]).cumprod()
            row[f"{name}_cagr"] = account_metrics(normalized)["cagr"]
        if row.get("return_alpha_dynamic_cagr") is not None:
            row["alpha_beats_no_margin"] = row["return_alpha_dynamic_cagr"] > row["no_margin_cagr"]
            row["alpha_beats_fixed_matched"] = row["return_alpha_dynamic_cagr"] > row["fixed_exposure_matched_cagr"]
        rows.append(row)
    return pd.DataFrame(rows)



def sensitivity_profiles(base: ReturnAlphaConfig) -> List[Tuple[str, ReturnAlphaConfig]]:
    """One-at-a-time neighborhood; avoids a costly in-sample grid search."""
    base_profile = copy.deepcopy(base)
    base_profile._sensitivity_rank_offset = 0.0
    base_profile._sensitivity_alpha_offset_pct = 0.0
    profiles: List[Tuple[str, ReturnAlphaConfig]] = [("base", base_profile)]

    for offset in base.sensitivity_rank_offsets:
        if abs(float(offset)) < 1e-12:
            continue
        cfg = copy.deepcopy(base)
        for name in (
            "rank_tier_1",
            "rank_tier_2",
            "rank_tier_3",
            "rank_tier_4",
        ):
            setattr(
                cfg,
                name,
                float(np.clip(getattr(base, name) + offset, 0.50, 0.995)),
            )
        cfg._sensitivity_rank_offset = float(offset)
        cfg._sensitivity_alpha_offset_pct = 0.0
        profiles.append((f"rank_{offset:+.2f}", cfg))

    for offset in base.sensitivity_alpha_offsets_pct:
        if abs(float(offset)) < 1e-12:
            continue
        cfg = copy.deepcopy(base)
        for name in (
            "relative_alpha_tier_1_pct",
            "relative_alpha_tier_2_pct",
            "relative_alpha_tier_3_pct",
            "relative_alpha_tier_4_pct",
        ):
            setattr(cfg, name, max(0.0, getattr(base, name) + offset))
        cfg._sensitivity_rank_offset = 0.0
        cfg._sensitivity_alpha_offset_pct = float(offset)
        profiles.append((f"alpha_{offset:+.2f}", cfg))

    for interval in base.sensitivity_decision_intervals:
        if int(interval) == int(base.decision_interval_business_days):
            continue
        cfg = copy.deepcopy(base)
        cfg.decision_interval_business_days = int(interval)
        cfg._sensitivity_rank_offset = 0.0
        cfg._sensitivity_alpha_offset_pct = 0.0
        profiles.append((f"interval_{int(interval)}", cfg))

    for max_margin in base.sensitivity_max_margins:
        if abs(float(max_margin) - float(base.max_alpha_margin_pct)) < 1e-12:
            continue
        cfg = copy.deepcopy(base)
        cfg.max_alpha_margin_pct = float(max_margin)
        scale = float(max_margin) / max(float(base.max_alpha_margin_pct), 1e-9)
        for name in (
            "margin_tier_1_pct",
            "margin_tier_2_pct",
            "margin_tier_3_pct",
            "margin_tier_4_pct",
        ):
            setattr(cfg, name, min(float(max_margin), getattr(base, name) * scale))
        cfg._sensitivity_rank_offset = 0.0
        cfg._sensitivity_alpha_offset_pct = 0.0
        profiles.append((f"max_margin_{float(max_margin):.0f}", cfg))

    # Preserve order while removing accidental duplicates.
    unique: List[Tuple[str, ReturnAlphaConfig]] = []
    seen: set = set()
    for name, cfg in profiles:
        key = (
            cfg.rank_tier_1,
            cfg.rank_tier_2,
            cfg.rank_tier_3,
            cfg.rank_tier_4,
            cfg.relative_alpha_tier_1_pct,
            cfg.relative_alpha_tier_2_pct,
            cfg.relative_alpha_tier_3_pct,
            cfg.relative_alpha_tier_4_pct,
            cfg.decision_interval_business_days,
            cfg.max_alpha_margin_pct,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, cfg))
    return unique

def run_parameter_sensitivity(
    history: pd.DataFrame,
    prices: pd.Series,
    forecasts: pd.DataFrame,
    sensitivity_cfg: ReturnAlphaConfig,
    alpha_state_cfg: StateMachineConfig,
    bt_cfg: BacktestConfig,
    evaluation_start: pd.Timestamp,
) -> pd.DataFrame:
    """Evaluate every declared neighborhood and always return explicit rows."""
    evaluation_history = history.loc[evaluation_start:].copy()
    evaluation_prices = prices.reindex(evaluation_history.index).dropna()
    evaluation_history = evaluation_history.reindex(evaluation_prices.index)

    no_margin_ledger = simulate_account(
        evaluation_prices,
        pd.Series(0.0, index=evaluation_history.index),
        bt_cfg,
        "no_margin",
    )
    no_margin_metrics = account_metrics(
        no_margin_ledger,
        bt_cfg.trading_days_per_year,
    )
    rows: List[Dict[str, Any]] = []
    profiles = sensitivity_profiles(sensitivity_cfg)
    if not profiles:
        raise RuntimeError("No sensitivity profiles were generated.")

    for profile_name, profile_cfg in profiles:
        raw_target, decisions = build_alpha_targets(
            history,
            forecasts,
            profile_cfg,
        )
        dynamic_target, _ = apply_state_machine(
            history,
            prices,
            forecasts,
            raw_target,
            decisions,
            profile_cfg,
            alpha_state_cfg,
        )
        evaluation_target = dynamic_target.reindex(
            evaluation_history.index
        ).fillna(0.0)
        alpha_ledger = simulate_account(
            evaluation_prices,
            evaluation_target,
            bt_cfg,
            "return_alpha_dynamic",
        )
        alpha_metrics = account_metrics(
            alpha_ledger,
            bt_cfg.trading_days_per_year,
        )
        average_exposure = float(
            alpha_ledger["effective_target_margin_pct"].mean()
        )
        matched_ledger = simulate_account(
            evaluation_prices,
            pd.Series(average_exposure, index=evaluation_history.index),
            bt_cfg,
            "fixed_exposure_matched",
        )
        matched_metrics = account_metrics(
            matched_ledger,
            bt_cfg.trading_days_per_year,
        )
        simple_ledgers = {
            "no_margin": no_margin_ledger,
            "fixed_exposure_matched": matched_ledger,
            "return_alpha_dynamic": alpha_ledger,
        }
        folds = fold_analysis(
            simple_ledgers,
            sensitivity_cfg.evaluation_fold_years,
        )
        valid = folds.dropna(subset=["return_alpha_dynamic_cagr"])
        rows.append({
            "profile": profile_name,
            "rank_offset": float(
                getattr(profile_cfg, "_sensitivity_rank_offset", 0.0)
            ),
            "relative_alpha_offset_pct": float(
                getattr(profile_cfg, "_sensitivity_alpha_offset_pct", 0.0)
            ),
            "decision_interval_business_days": profile_cfg.decision_interval_business_days,
            "max_alpha_margin_pct": profile_cfg.max_alpha_margin_pct,
            "alpha_cagr": alpha_metrics["cagr"],
            "fixed_matched_cagr": matched_metrics["cagr"],
            "no_margin_cagr": no_margin_metrics["cagr"],
            "alpha_minus_fixed_matched_cagr": (
                float(alpha_metrics["cagr"])
                - float(matched_metrics["cagr"])
            ),
            "alpha_minus_no_margin_cagr": (
                float(alpha_metrics["cagr"])
                - float(no_margin_metrics["cagr"])
            ),
            "alpha_max_drawdown": alpha_metrics["max_drawdown"],
            "average_target_margin_pct": alpha_metrics["average_target_margin_pct"],
            "annual_trades": alpha_metrics["annual_trades"],
            "fold_win_rate_vs_fixed_matched": (
                float(valid["alpha_beats_fixed_matched"].mean())
                if not valid.empty
                else None
            ),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Sensitivity output is unexpectedly empty.")
    return frame


def run_full_validation(
    history: pd.DataFrame,
    prices: pd.Series,
    alpha_cfg: ReturnAlphaConfig,
    alpha_state_cfg: StateMachineConfig,
    current_state_cfg: StateMachineConfig,
    bt_cfg: BacktestConfig,
    sensitivity_cfg: Optional[ReturnAlphaConfig] = None,
) -> Dict[str, Any]:
    common = history.index.intersection(prices.index)
    history = history.reindex(common)
    prices = prices.reindex(common)
    forecasts, model_records = generate_oos_forecasts(history, prices, alpha_cfg)
    alpha_raw, alpha_records = build_alpha_targets(history, forecasts, alpha_cfg)
    alpha_dynamic, state_records = apply_state_machine(
        history,
        prices,
        forecasts,
        alpha_raw,
        alpha_records,
        alpha_cfg,
        alpha_state_cfg,
    )

    evaluation_start = forecasts["expected_net_return_60d"].first_valid_index()
    if evaluation_start is None:
        raise RuntimeError("No out-of-sample return forecasts were generated.")

    ledgers, metrics_frame = compare_strategies(
        history,
        prices,
        alpha_raw,
        alpha_dynamic,
        current_state_cfg,
        bt_cfg,
        evaluation_start,
    )

    stress_cfg = copy.deepcopy(bt_cfg)
    stress_cfg.annual_borrow_rate += 0.02
    stress_cfg.transaction_cost_bps *= 2.0
    stress_cfg.slippage_bps *= 2.0
    stress_ledgers, stress_metrics = compare_strategies(
        history,
        prices,
        alpha_raw,
        alpha_dynamic,
        current_state_cfg,
        stress_cfg,
        evaluation_start,
    )

    alpha_return = ledgers["return_alpha_dynamic"]["strategy_return"]
    no_margin_return = ledgers["no_margin"]["strategy_return"]
    matched_return = ledgers["fixed_exposure_matched"]["strategy_return"]
    daily_diff = alpha_return - matched_return

    statistics = {
        "newey_west_vs_fixed_matched": newey_west_t_stat(daily_diff, 10),
        "monthly_block_bootstrap_vs_fixed_matched": block_bootstrap_monthly_difference(
            alpha_return,
            matched_return,
            alpha_cfg.bootstrap_iterations,
            alpha_cfg.bootstrap_block_months,
            alpha_cfg.random_seed,
        ),
        "monthly_block_bootstrap_vs_no_margin": block_bootstrap_monthly_difference(
            alpha_return,
            no_margin_return,
            alpha_cfg.bootstrap_iterations,
            alpha_cfg.bootstrap_block_months,
            alpha_cfg.random_seed + 1,
        ),
        "forecast_diagnostics": forecast_diagnostics(
            prices.loc[evaluation_start:],
            alpha_dynamic.loc[evaluation_start:],
            60,
        ),
        "rank_diagnostics": rank_forecast_diagnostics(
            forecasts.loc[evaluation_start:],
            prices.loc[evaluation_start:],
            alpha_records,
            alpha_cfg,
        ),
    }
    folds = fold_analysis(ledgers, alpha_cfg.evaluation_fold_years)
    sensitivity = (
        run_parameter_sensitivity(
            history,
            prices,
            forecasts,
            sensitivity_cfg,
            alpha_state_cfg,
            bt_cfg,
            evaluation_start,
        )
        if sensitivity_cfg is not None
        else pd.DataFrame()
    )

    pit_series = history.get(
        "historical_data_is_revised",
        pd.Series(True, index=history.index),
    )
    pit_ready = not pit_series.map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"}
    ).any()
    return {
        "history": history,
        "prices": prices,
        "forecasts": forecasts,
        "model_records": model_records,
        "alpha_records": alpha_records,
        "state_records": state_records,
        "alpha_raw": alpha_raw,
        "alpha_dynamic": alpha_dynamic,
        "ledgers": ledgers,
        "stress_ledgers": stress_ledgers,
        "metrics": metrics_frame,
        "stress_metrics": stress_metrics,
        "statistics": statistics,
        "folds": folds,
        "sensitivity": sensitivity,
        "pit_ready": pit_ready,
        "evaluation_start": evaluation_start,
        "alpha_cfg_version": alpha_cfg.model_version,
    }


def validation_summary(result: Mapping[str, Any]) -> Dict[str, Any]:
    metrics_frame = result["metrics"].set_index("strategy")
    stress_frame = result["stress_metrics"].set_index("strategy")
    alpha = metrics_frame.loc["return_alpha_dynamic"]
    no_margin = metrics_frame.loc["no_margin"]
    matched = metrics_frame.loc["fixed_exposure_matched"]
    stress_alpha = stress_frame.loc["return_alpha_dynamic"]
    stress_matched = stress_frame.loc["fixed_exposure_matched"]
    folds = result["folds"]
    valid_folds = folds.dropna(subset=["return_alpha_dynamic_cagr"])
    bootstrap = result["statistics"]["monthly_block_bootstrap_vs_fixed_matched"]
    rank_diag = result["statistics"].get("rank_diagnostics", {})
    sensitivity = result["sensitivity"]
    sensitivity_positive_rate = (
        float((sensitivity["alpha_minus_fixed_matched_cagr"] > 0).mean())
        if not sensitivity.empty
        else 0.0
    )
    sensitivity_target_rate = (
        float((sensitivity["alpha_minus_fixed_matched_cagr"] >= 0.005).mean())
        if not sensitivity.empty
        else 0.0
    )
    excess_cagr = float(alpha["cagr"]) - float(matched["cagr"])
    stress_excess_cagr = float(stress_alpha["cagr"]) - float(stress_matched["cagr"])
    fold_win_rate = (
        float(valid_folds["alpha_beats_fixed_matched"].mean())
        if not valid_folds.empty
        else 0.0
    )
    rank_ic = rank_diag.get("spearman_rank_ic")
    top_relative = rank_diag.get("top_decile_realized_relative_alpha")

    checks = {
        "alpha_cagr_above_no_margin": float(alpha["cagr"]) > float(no_margin["cagr"]),
        "fixed_matched_excess_cagr_above_0_50pp": excess_cagr >= 0.005,
        "alpha_drawdown_not_worse_than_no_margin_by_5pp": (
            float(alpha["max_drawdown"])
            >= float(no_margin["max_drawdown"]) - 0.05
        ),
        "alpha_annual_trades_below_25": float(alpha["annual_trades"]) <= 25.0,
        "bootstrap_probability_positive_above_70pct": (
            float(bootstrap.get("probability_positive") or 0.0) >= 0.70
        ),
        "fold_win_rate_vs_fixed_matched_above_60pct": fold_win_rate >= 0.60,
        "stress_excess_cagr_positive": stress_excess_cagr > 0.0,
        "rank_ic_above_0_05": rank_ic is not None and float(rank_ic) >= 0.05,
        "top_decile_realized_relative_alpha_positive": (
            top_relative is not None and float(top_relative) > 0.0
        ),
    }
    if not sensitivity.empty:
        checks["sensitivity_0_50pp_excess_rate_above_60pct"] = (
            sensitivity_target_rate >= 0.60
        )

    return {
        "model_version": result.get("alpha_cfg_version", "unknown"),
        "strict_pit_ready": bool(result["pit_ready"]),
        "fixed_matched_excess_cagr": excess_cagr,
        "stress_fixed_matched_excess_cagr": stress_excess_cagr,
        "fold_win_rate_vs_fixed_matched": fold_win_rate,
        "rank_ic": rank_ic,
        "sensitivity_profile_count": int(len(sensitivity)),
        "sensitivity_positive_rate": sensitivity_positive_rate,
        "sensitivity_0_50pp_excess_rate": sensitivity_target_rate,
        "checks": checks,
        "passed_count": int(sum(checks.values())),
        "total_count": len(checks),
        "conclusion": (
            "PASS"
            if all(checks.values()) and result["pit_ready"]
            else "PROMISING_BUT_REQUIRES_STRICT_PIT"
            if sum(checks.values()) >= max(7, len(checks) - 2)
            and not result["pit_ready"]
            else "INSUFFICIENT_RETURN_TIMING_EVIDENCE"
        ),
    }


def write_validation_outputs(output_dir: Path, result: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result["forecasts"].to_csv(output_dir / "oos_return_forecasts.csv")
    result["model_records"].to_csv(output_dir / "model_refit_audit.csv", index=False)
    result["alpha_records"].to_csv(output_dir / "alpha_target_decisions.csv", index=False)
    result["state_records"].to_csv(output_dir / "alpha_state_machine_decisions.csv", index=False)
    result["metrics"].to_csv(output_dir / "strategy_comparison.csv", index=False)
    result["stress_metrics"].to_csv(output_dir / "strategy_comparison_stress.csv", index=False)
    result["folds"].to_csv(output_dir / "walk_forward_folds.csv", index=False)
    result["sensitivity"].to_csv(
        output_dir / "parameter_sensitivity.csv",
        index=False,
    )
    for name, ledger in result["ledgers"].items():
        ledger.to_csv(output_dir / f"ledger_{name}.csv")
    summary = validation_summary(result)
    report = {
        "pit_status": {
            "strict_pit_ready": result["pit_ready"],
            "message": (
                "Strict PIT history detected."
                if result["pit_ready"]
                else "Exploratory only: score history contains revised/live_public data."
            ),
        },
        "summary": summary,
        "evaluation_start": result["evaluation_start"].date().isoformat(),
        "statistics": result["statistics"],
        "strategy_metrics": result["metrics"].to_dict("records"),
        "stress_metrics": result["stress_metrics"].to_dict("records"),
        "walk_forward_folds": result["folds"].to_dict("records"),
        "parameter_sensitivity": result["sensitivity"].to_dict("records"),
    }
    save_json(output_dir / "return_alpha_validation_report.json", report)


def load_history_and_price(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.Series]:
    history = load_score_history(Path(args.history))
    prices = load_price_series(
        args.ticker,
        history.index.min(),
        history.index.max(),
        args.price_csv,
    )
    common = history.index.intersection(prices.index)
    if len(common) < 500:
        raise ValueError("Fewer than 500 common history and price rows.")
    return history.reindex(common), prices.reindex(common)


def command_backtest(args: argparse.Namespace) -> None:
    history, prices = load_history_and_price(args)
    alpha_cfg = load_dataclass_config(args.alpha_config, ReturnAlphaConfig)
    alpha_cfg.ticker = args.ticker
    alpha_state_cfg = load_existing_config(args.alpha_state_config, StateMachineConfig)
    current_state_cfg = load_existing_config(args.current_state_config, StateMachineConfig)
    bt_cfg = load_existing_config(args.backtest_config, BacktestConfig)
    bt_cfg.ticker = args.ticker
    sensitivity_cfg = (
        load_dataclass_config(args.sensitivity_config, ReturnAlphaConfig)
        if getattr(args, "sensitivity_config", None)
        else None
    )
    if sensitivity_cfg is not None:
        sensitivity_cfg.ticker = args.ticker
    result = run_full_validation(
        history,
        prices,
        alpha_cfg,
        alpha_state_cfg,
        current_state_cfg,
        bt_cfg,
        sensitivity_cfg=sensitivity_cfg,
    )
    write_validation_outputs(Path(args.output_dir), result)
    print(result["metrics"].to_string(index=False))
    print(json.dumps(validation_summary(result), ensure_ascii=False, indent=2))


def command_live(args: argparse.Namespace) -> None:
    history, prices = load_history_and_price(args)
    alpha_cfg = load_dataclass_config(args.alpha_config, ReturnAlphaConfig)
    alpha_cfg.ticker = args.ticker
    alpha_state_cfg = load_existing_config(
        args.alpha_state_config,
        StateMachineConfig,
    )

    latest_forecast, _ = generate_latest_forecast(
        history,
        prices,
        alpha_cfg,
    )
    latest_date = history.index[-1]
    latest_row = history.loc[latest_date]
    signal_payload = json.loads(Path(args.signals).read_text(encoding="utf-8"))
    state = load_margin_state(Path(args.state), args.initial_actual_margin_pct)

    decision_due = (
        state.last_model_decision_date is None
        or business_days_between_dates(
            state.last_model_decision_date,
            latest_date,
        ) >= alpha_cfg.decision_interval_business_days
    )
    if decision_due:
        demand, alpha_reasons = alpha_demand_from_forecast(
            latest_forecast,
            alpha_cfg,
        )
    else:
        demand = float(state.scheduled_alpha_demand_margin_pct)
        alpha_reasons = ["Carry_Previous_Scheduled_Demand"]

    risk_cap, risk_reasons = risk_cap_from_scores(latest_row, alpha_cfg)
    raw_target = min(demand, risk_cap)
    new_state, decision = update_alpha_state(
        latest_date,
        float(prices.loc[latest_date]),
        raw_target,
        latest_forecast,
        latest_row,
        state,
        alpha_cfg,
        alpha_state_cfg,
        decision_due=decision_due,
        scheduled_demand=demand,
    )
    save_json(Path(args.state), asdict(new_state))

    reference_equity = float(args.reference_equity)
    execution = {
        "model_version": alpha_cfg.model_version,
        "signal_date": latest_date.date().isoformat(),
        "effective_date": signal_payload.get(
            "decision_date",
            latest_date.date().isoformat(),
        ),
        "decision_due": bool(decision_due),
        "decision_interval_business_days": alpha_cfg.decision_interval_business_days,
        "last_model_decision_date": new_state.last_model_decision_date,
        "expert": latest_forecast.get("expert"),
        "expected_relative_alpha_60d_pct": (
            100.0 * float(latest_forecast["expected_relative_alpha_60d"])
        ),
        "stable_expected_relative_alpha_60d_pct": (
            100.0 * float(
                latest_forecast["stable_expected_relative_alpha_60d"]
            )
        ),
        "static_baseline_60d_pct": (
            100.0 * float(latest_forecast["static_baseline_60d"])
        ),
        "expected_net_return_60d_pct": (
            100.0 * float(latest_forecast["expected_net_return_60d"])
        ),
        "forecast_rank": float(latest_forecast["forecast_rank"]),
        "stable_forecast_rank": float(
            latest_forecast["stable_forecast_rank"]
        ),
        "rank_pass_count": int(round(float(latest_forecast["rank_pass_count"]))),
        "rank_stability_lookback_days": alpha_cfg.rank_stability_lookback_days,
        "diagnostic_probability": float(
            latest_forecast["diagnostic_probability"]
        ),
        "alpha_demand_margin_pct": demand,
        "risk_cap_margin_pct": risk_cap,
        "alpha_reasons": alpha_reasons,
        "risk_reasons": risk_reasons,
        "raw_target_margin_pct": raw_target,
        "confirmed_target_margin_pct": decision["confirmed_target_margin_pct"],
        "model_actual_margin_pct": decision["model_actual_margin_pct"],
        "action": decision["action"],
        "reasons": decision["reasons"],
        "reference_equity": reference_equity,
        "recommended_loan_amount": (
            reference_equity
            * float(decision["model_actual_margin_pct"])
            / 100.0
        ),
        "entry_date": decision.get("entry_date"),
        "entry_price": decision.get("entry_price"),
        "peak_price": decision.get("peak_price"),
        "model_train_end": latest_forecast.get("model_train_end"),
        "strict_pit_ready": not history.get(
            "historical_data_is_revised",
            pd.Series(True, index=history.index),
        ).map(
            lambda value: str(value).strip().lower()
            in {"1", "true", "yes", "y"}
        ).any(),
        "note": (
            "V3 uses direct relative-alpha ranking, a 10-business-day "
            "decision cadence, and daily risk exits. Diagnostic probability "
            "does not determine leverage."
        ),
    }
    save_json(Path(args.execution_output), execution)
    signal_payload["return_alpha"] = execution
    save_json(Path(args.signals), signal_payload)
    print(json.dumps(json_safe(execution), ensure_ascii=False, indent=2))


def command_self_test(_: argparse.Namespace) -> None:
    rng = np.random.default_rng(415)
    dates = pd.bdate_range("2000-01-03", periods=2200)
    regime = np.sin(np.arange(len(dates)) / 90.0)
    pullback = -0.03 * (np.sin(np.arange(len(dates)) / 17.0) > 0.85)
    returns = (
        0.00020
        + 0.00055 * regime
        - 0.00020 * (regime < -0.4)
        + rng.normal(0, 0.009, len(dates))
    )
    prices = pd.Series(100.0 * np.cumprod(1.0 + returns), index=dates)
    history = pd.DataFrame(index=dates)
    history["Market_Regime"] = 50 + 30 * regime
    history["AI_Cycle"] = 60.0
    history["Valuation"] = 50.0
    history["Macro"] = 50.0
    history["Liquidity"] = 55 + 10 * regime
    history["Positioning"] = 50.0
    history["Buy_Score"] = 55 + 20 * regime
    history["Risk_Score"] = 45 - 20 * regime
    history["Margin_Score"] = 55 + 15 * regime
    history["Coverage_Ratio"] = 1.0
    history["HY_OAS_Percentile"] = 40 - 15 * regime
    history["Breadth_Score"] = 50 + 25 * regime
    history["VIX"] = 20 - 5 * regime
    history["VIX3M"] = history["VIX"] + 2.0
    history["VIX_Backwardation"] = 0.0
    history["NFL_13W_Change"] = 10 * regime
    history["Net_Fed_Liquidity"] = 5000 + np.cumsum(regime)
    history["Target_Margin_Pct"] = np.where(regime > 0, 5.0, 0.0)
    history["historical_data_is_revised"] = False

    cfg = ReturnAlphaConfig(
        horizons=(20, 40),
        horizon_weights=(0.5, 0.5),
        min_train_days=600,
        train_window_days=1200,
        bootstrap_iterations=100,
    )
    state_cfg = StateMachineConfig(
        max_target_margin_pct=8.0,
        increase_confirm_days=2,
        decrease_confirm_days=1,
        max_increase_step_pct=2.0,
        max_decrease_step_pct=8.0,
    )
    bt_cfg = BacktestConfig()
    result = run_full_validation(
        history,
        prices,
        cfg,
        state_cfg,
        StateMachineConfig(),
        bt_cfg,
    )
    assert result["forecasts"]["expected_relative_alpha_60d"].notna().sum() > 100
    assert result["forecasts"]["stable_forecast_rank"].dropna().between(0, 1).all()
    assert result["alpha_dynamic"].max() <= cfg.max_alpha_margin_pct
    assert "fixed_exposure_matched" in result["ledgers"]

    altered = prices.copy()
    altered.iloc[-100:] *= np.linspace(1.0, 1.5, 100)
    forecast_a, _ = generate_oos_forecasts(history, prices, cfg)
    forecast_b, _ = generate_oos_forecasts(history, altered, cfg)
    safe_end = dates[-100 - max(cfg.horizons) - 5]
    comparison = pd.concat(
        [
            forecast_a.loc[:safe_end, "expected_relative_alpha_60d"],
            forecast_b.loc[:safe_end, "expected_relative_alpha_60d"],
        ],
        axis=1,
    ).dropna()
    assert np.allclose(comparison.iloc[:, 0], comparison.iloc[:, 1], atol=1e-12)
    print("Return Alpha V3 self-test passed.")
    print(result["metrics"].to_string(index=False))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Return-seeking margin alpha engine and validation.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--history", default="output/score_history.csv")
        command.add_argument("--ticker", default="QQQ")
        command.add_argument("--price-csv", default=None)
        command.add_argument("--alpha-config", default="validation/return_alpha_config.json")
        command.add_argument("--alpha-state-config", default="validation/return_alpha_state_config.json")
        command.add_argument("--current-state-config", default="validation/margin_state_config.json")
        command.add_argument("--backtest-config", default="validation/margin_backtest_config.json")

    backtest = sub.add_parser("backtest")
    common(backtest)
    backtest.add_argument("--output-dir", default="output/return_alpha_validation")
    backtest.add_argument("--sensitivity-config", default=None)
    backtest.set_defaults(func=command_backtest)

    live = sub.add_parser("live")
    common(live)
    live.add_argument("--signals", default="output/latest_signals.json")
    live.add_argument("--state", default="output/return_alpha_state.json")
    live.add_argument("--execution-output", default="output/return_alpha_execution.json")
    live.add_argument("--reference-equity", type=float, required=True)
    live.add_argument("--initial-actual-margin-pct", type=float, default=0.0)
    live.set_defaults(func=command_live)

    test = sub.add_parser("self-test")
    test.set_defaults(func=command_self_test)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
