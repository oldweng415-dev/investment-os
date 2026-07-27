#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

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
    ticker: str = "QQQ"
    horizons: Tuple[int, ...] = (20, 60, 120)
    horizon_weights: Tuple[float, ...] = (0.20, 0.55, 0.25)
    min_train_days: int = 1260
    train_window_days: int = 2520
    refit_frequency: str = "monthly"
    ridge_penalty: float = 25.0
    target_winsor_lower: float = 0.01
    target_winsor_upper: float = 0.99
    annual_borrow_rate: float = 0.04
    round_trip_cost_bps: float = 14.0

    probability_tier_1: float = 0.54
    probability_tier_2: float = 0.58
    probability_tier_3: float = 0.62
    probability_tier_4: float = 0.66
    expected_net_60d_tier_1_pct: float = 0.50
    expected_net_60d_tier_2_pct: float = 1.50
    expected_net_60d_tier_3_pct: float = 3.00
    expected_net_60d_tier_4_pct: float = 5.00
    margin_tier_1_pct: float = 1.0
    margin_tier_2_pct: float = 3.0
    margin_tier_3_pct: float = 5.0
    margin_tier_4_pct: float = 8.0
    max_alpha_margin_pct: float = 8.0

    require_price_above_ma200: bool = True
    pullback_bonus_pct: float = 1.0
    pullback_min_pct: float = -12.0
    pullback_max_pct: float = -2.0
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

    evaluation_fold_years: int = 2
    bootstrap_iterations: int = 2000
    bootstrap_block_months: int = 3
    random_seed: int = 415
    run_sensitivity: bool = False
    sensitivity_probability_offsets: Tuple[float, ...] = (-0.02, 0.0, 0.02)
    sensitivity_max_margins: Tuple[float, ...] = (5.0, 8.0, 10.0)


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
            "sensitivity_probability_offsets",
            "sensitivity_max_margins",
        ):
            if name in values:
                values[name] = tuple(values[name])
    return cls(**values)



def load_margin_state(path: Path, initial_actual_margin_pct: float = 0.0) -> MarginState:
    if not path.exists():
        return MarginState(
            model_actual_margin_pct=float(initial_actual_margin_pct),
            confirmed_target_margin_pct=float(initial_actual_margin_pct),
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(MarginState)}
    clean = {key: value for key, value in payload.items() if key in allowed}
    return MarginState(**clean)

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

    for days in (5, 20, 60, 120, 252):
        feature[f"momentum_{days}d"] = price.pct_change(days)

    for days in (20, 50, 100, 200):
        ma = price.rolling(days, min_periods=days).mean()
        feature[f"price_to_ma{days}"] = price / ma - 1.0

    ma50 = price.rolling(50, min_periods=50).mean()
    ma200 = price.rolling(200, min_periods=200).mean()
    feature["ma50_to_ma200"] = ma50 / ma200 - 1.0

    for days in (20, 60, 120):
        feature[f"realized_vol_{days}d"] = returns.rolling(days, min_periods=days).std() * math.sqrt(252)

    for days in (20, 60, 252):
        rolling_high = price.rolling(days, min_periods=days).max()
        feature[f"drawdown_{days}d"] = price / rolling_high - 1.0

    feature["rsi14"] = rsi(price, 14) / 100.0
    feature["vix_term_spread"] = (
        pd.to_numeric(history.get("VIX3M"), errors="coerce")
        - pd.to_numeric(history.get("VIX"), errors="coerce")
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
    ]
    for column in change_columns:
        if column in feature.columns:
            feature[f"{column}_change_5d"] = feature[column].diff(5)
            feature[f"{column}_change_20d"] = feature[column].diff(20)

    feature = feature.replace([np.inf, -np.inf], np.nan)
    return feature


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


def fit_ridge_forecaster(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_predict: pd.DataFrame,
    ridge_penalty: float,
    lower_quantile: float,
    upper_quantile: float,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    valid = y_train.notna()
    x_train = x_train.loc[valid]
    y_train = y_train.loc[valid].astype(float)
    if len(y_train) < 250:
        raise ValueError("Fewer than 250 valid labeled observations.")

    lower = float(y_train.quantile(lower_quantile))
    upper = float(y_train.quantile(upper_quantile))
    y = y_train.clip(lower, upper).to_numpy(float)

    medians = x_train.median(axis=0, skipna=True).fillna(0.0)
    train = x_train.fillna(medians).astype(float)
    predict = x_predict.fillna(medians).astype(float)

    means = train.mean(axis=0)
    stds = train.std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
    train_z = ((train - means) / stds).clip(-8, 8)
    predict_z = ((predict - means) / stds).clip(-8, 8)

    x = np.column_stack([np.ones(len(train_z)), train_z.to_numpy(float)])
    xp = np.column_stack([np.ones(len(predict_z)), predict_z.to_numpy(float)])
    penalty = np.eye(x.shape[1]) * ridge_penalty
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)

    fitted = x @ beta
    residual = y - fitted
    residual_std = max(float(np.std(residual, ddof=1)), 1e-6)
    prediction = np.clip(xp @ beta, -0.50, 0.80)
    probability = norm.cdf(prediction / residual_std)

    pred_series = pd.Series(prediction, index=x_predict.index)
    prob_series = pd.Series(probability, index=x_predict.index).clip(0.01, 0.99)
    diagnostics = {
        "train_rows": int(len(y_train)),
        "residual_std": residual_std,
        "target_mean": float(np.mean(y)),
        "target_positive_rate": float(np.mean(y > 0)),
        "feature_count": int(train_z.shape[1]),
    }
    return pred_series, prob_series, diagnostics


def monthly_refit_dates(index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    frame = pd.DataFrame(index=index)
    return list(frame.groupby([index.year, index.month]).head(1).index)


def generate_oos_forecasts(
    history: pd.DataFrame,
    prices: pd.Series,
    cfg: ReturnAlphaConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate monthly-refitted, purged out-of-sample forecasts.

    Each horizon uses its own label embargo. For example, a 20-day model
    predicting on date t may train only through t-21, while a 120-day model
    may train only through t-121. This avoids wasting recent labels for the
    shorter horizon without leaking incomplete outcomes.
    """

    features = build_features(history, prices)
    prices = prices.reindex(features.index)
    targets = future_net_returns(prices, cfg)

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

        horizon_predictions: List[pd.Series] = []
        horizon_probabilities: List[pd.Series] = []
        horizon_train_ends: List[pd.Timestamp] = []
        all_horizons_available = True

        for horizon in cfg.horizons:
            label_end_position = start_position - horizon - 1
            if label_end_position < cfg.min_train_days - 1:
                all_horizons_available = False
                break

            train_start_position = max(
                0,
                label_end_position - cfg.train_window_days + 1,
            )
            train_index = features.index[
                train_start_position:
                label_end_position + 1
            ]

            pred, prob, diagnostics = fit_ridge_forecaster(
                features.loc[train_index],
                targets[horizon].loc[train_index],
                features.loc[prediction_index],
                cfg.ridge_penalty,
                cfg.target_winsor_lower,
                cfg.target_winsor_upper,
            )

            # Normalize every horizon to an approximate 60-trading-day scale.
            scaled_pred = pred * (60.0 / horizon)
            horizon_predictions.append(scaled_pred)
            horizon_probabilities.append(prob)
            horizon_train_ends.append(train_index[-1])

            model_records.append(
                {
                    "prediction_start": prediction_start.date().isoformat(),
                    "prediction_end": prediction_index[-1].date().isoformat(),
                    "train_start": train_index[0].date().isoformat(),
                    "train_end": train_index[-1].date().isoformat(),
                    "purge_trading_days": horizon + 1,
                    "horizon_days": horizon,
                    **diagnostics,
                }
            )

        if not all_horizons_available:
            continue

        weights = np.asarray(cfg.horizon_weights, dtype=float)
        weights = weights / weights.sum()
        pred_frame = pd.concat(horizon_predictions, axis=1)
        prob_frame = pd.concat(horizon_probabilities, axis=1)

        output.loc[
            prediction_index,
            "expected_net_return_60d",
        ] = pred_frame.mul(weights, axis=1).sum(axis=1)
        output.loc[
            prediction_index,
            "probability_positive",
        ] = prob_frame.mul(weights, axis=1).sum(axis=1)
        output.loc[
            prediction_index,
            "model_train_end",
        ] = max(horizon_train_ends).date().isoformat()

    output["price_to_ma200"] = features["price_to_ma200"]
    output["ma50_to_ma200"] = features["ma50_to_ma200"]
    output["drawdown_60d"] = features["drawdown_60d"]
    output["momentum_20d"] = features["momentum_20d"]
    return output, pd.DataFrame(model_records)


def generate_latest_forecast(
    history: pd.DataFrame,
    prices: pd.Series,
    cfg: ReturnAlphaConfig,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Fit only the models needed for the latest live decision."""

    features = build_features(history, prices)
    prices = prices.reindex(features.index)
    targets = future_net_returns(prices, cfg)
    latest_date = features.index[-1]
    latest_position = len(features.index) - 1

    predictions: List[float] = []
    probabilities: List[float] = []
    records: List[Dict[str, Any]] = []
    train_ends: List[pd.Timestamp] = []

    for horizon in cfg.horizons:
        label_end_position = latest_position - horizon - 1
        if label_end_position < cfg.min_train_days - 1:
            raise RuntimeError(
                f"Insufficient labeled history for {horizon}-day live model."
            )

        train_start_position = max(
            0,
            label_end_position - cfg.train_window_days + 1,
        )
        train_index = features.index[
            train_start_position:
            label_end_position + 1
        ]

        pred, prob, diagnostics = fit_ridge_forecaster(
            features.loc[train_index],
            targets[horizon].loc[train_index],
            features.loc[[latest_date]],
            cfg.ridge_penalty,
            cfg.target_winsor_lower,
            cfg.target_winsor_upper,
        )

        predictions.append(float(pred.iloc[0]) * 60.0 / horizon)
        probabilities.append(float(prob.iloc[0]))
        train_ends.append(train_index[-1])
        records.append(
            {
                "prediction_date": latest_date.date().isoformat(),
                "train_start": train_index[0].date().isoformat(),
                "train_end": train_index[-1].date().isoformat(),
                "purge_trading_days": horizon + 1,
                "horizon_days": horizon,
                **diagnostics,
            }
        )

    weights = np.asarray(cfg.horizon_weights, dtype=float)
    weights = weights / weights.sum()
    row = pd.Series(
        {
            "expected_net_return_60d": float(np.dot(predictions, weights)),
            "probability_positive": float(np.dot(probabilities, weights)),
            "model_train_end": max(train_ends).date().isoformat(),
            "price_to_ma200": float(features.loc[latest_date, "price_to_ma200"]),
            "ma50_to_ma200": float(features.loc[latest_date, "ma50_to_ma200"]),
            "drawdown_60d": float(features.loc[latest_date, "drawdown_60d"]),
            "momentum_20d": float(features.loc[latest_date, "momentum_20d"]),
        },
        name=latest_date,
    )
    return row, pd.DataFrame(records)


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
    expected = forecast_row.get("expected_net_return_60d")
    probability = forecast_row.get("probability_positive")
    if pd.isna(expected) or pd.isna(probability):
        return 0.0, ["Forecast_Unavailable"]

    expected_pct = 100.0 * float(expected)
    probability = float(probability)
    demand = 0.0
    tier = "No_Alpha_Tier"

    tiers = [
        (cfg.probability_tier_1, cfg.expected_net_60d_tier_1_pct, cfg.margin_tier_1_pct, "Tier_1"),
        (cfg.probability_tier_2, cfg.expected_net_60d_tier_2_pct, cfg.margin_tier_2_pct, "Tier_2"),
        (cfg.probability_tier_3, cfg.expected_net_60d_tier_3_pct, cfg.margin_tier_3_pct, "Tier_3"),
        (cfg.probability_tier_4, cfg.expected_net_60d_tier_4_pct, cfg.margin_tier_4_pct, "Tier_4"),
    ]
    for p_threshold, r_threshold, margin, name in tiers:
        if probability >= p_threshold and expected_pct >= r_threshold:
            demand = margin
            tier = name

    reasons = [tier]
    above_ma200 = float(forecast_row.get("price_to_ma200", -1.0)) > 0
    ma_trend = float(forecast_row.get("ma50_to_ma200", -1.0)) > 0
    if cfg.require_price_above_ma200 and not (above_ma200 and ma_trend):
        return 0.0, reasons + ["Long_Trend_Filter_Failed"]

    drawdown_pct = 100.0 * float(forecast_row.get("drawdown_60d", 0.0))
    if cfg.pullback_min_pct <= drawdown_pct <= cfg.pullback_max_pct and demand > 0:
        demand += cfg.pullback_bonus_pct
        reasons.append("Uptrend_Pullback_Bonus")

    return min(demand, cfg.max_alpha_margin_pct), reasons


def build_alpha_targets(
    history: pd.DataFrame,
    forecasts: pd.DataFrame,
    cfg: ReturnAlphaConfig,
) -> Tuple[pd.Series, pd.DataFrame]:
    raw_targets: Dict[pd.Timestamp, float] = {}
    records: List[Dict[str, Any]] = []
    for date in history.index.intersection(forecasts.index):
        forecast_row = forecasts.loc[date]
        history_row = history.loc[date]
        demand, alpha_reasons = alpha_demand_from_forecast(forecast_row, cfg)
        risk_cap, risk_reasons = risk_cap_from_scores(history_row, cfg)
        target = min(demand, risk_cap)
        raw_targets[date] = target
        records.append(
            {
                "date": date.date().isoformat(),
                "expected_net_return_60d_pct": (
                    100.0 * float(forecast_row["expected_net_return_60d"])
                    if pd.notna(forecast_row.get("expected_net_return_60d")) else None
                ),
                "probability_positive": (
                    float(forecast_row["probability_positive"])
                    if pd.notna(forecast_row.get("probability_positive")) else None
                ),
                "alpha_demand_margin_pct": demand,
                "risk_cap_margin_pct": risk_cap,
                "alpha_raw_target_margin_pct": target,
                "alpha_reasons": alpha_reasons,
                "risk_reasons": risk_reasons,
                "model_train_end": forecast_row.get("model_train_end"),
            }
        )
    return pd.Series(raw_targets, name="Alpha_Raw_Target_Margin_Pct").sort_index(), pd.DataFrame(records)


def apply_state_machine(
    history: pd.DataFrame,
    raw_target: pd.Series,
    state_cfg: StateMachineConfig,
) -> Tuple[pd.Series, pd.DataFrame]:
    state = MarginState()
    targets: Dict[pd.Timestamp, float] = {}
    records: List[Dict[str, Any]] = []
    for date in history.index:
        row = history.loc[date]
        signal = signal_from_history_row(date, row)
        signal["Target_Margin_Pct"] = float(raw_target.get(date, 0.0))
        state, decision = update_margin_state(signal, state, state_cfg)
        targets[date] = state.model_actual_margin_pct
        records.append({"date": date.date().isoformat(), **decision})
    return pd.Series(targets, name="Alpha_Dynamic_Target_Margin_Pct"), pd.DataFrame(records)


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
    """Create a small neighborhood around the base mapping.

    These profiles are reported together; the engine never silently selects
    the best historical profile. This is a robustness test, not a tuner.
    """

    profiles: List[Tuple[str, ReturnAlphaConfig]] = []
    for probability_offset in base.sensitivity_probability_offsets:
        for max_margin in base.sensitivity_max_margins:
            cfg = copy.deepcopy(base)
            cfg.probability_tier_1 = float(np.clip(base.probability_tier_1 + probability_offset, 0.50, 0.80))
            cfg.probability_tier_2 = float(np.clip(base.probability_tier_2 + probability_offset, 0.50, 0.80))
            cfg.probability_tier_3 = float(np.clip(base.probability_tier_3 + probability_offset, 0.50, 0.80))
            cfg.probability_tier_4 = float(np.clip(base.probability_tier_4 + probability_offset, 0.50, 0.80))
            cfg.max_alpha_margin_pct = max_margin
            scale = max_margin / max(base.max_alpha_margin_pct, 1e-9)
            cfg.margin_tier_1_pct = min(max_margin, base.margin_tier_1_pct * scale)
            cfg.margin_tier_2_pct = min(max_margin, base.margin_tier_2_pct * scale)
            cfg.margin_tier_3_pct = min(max_margin, base.margin_tier_3_pct * scale)
            cfg.margin_tier_4_pct = min(max_margin, base.margin_tier_4_pct * scale)
            name = f"p_offset_{probability_offset:+.2f}_max_{max_margin:.0f}"
            profiles.append((name, cfg))
    return profiles


def run_parameter_sensitivity(
    history: pd.DataFrame,
    prices: pd.Series,
    forecasts: pd.DataFrame,
    base_cfg: ReturnAlphaConfig,
    alpha_state_cfg: StateMachineConfig,
    bt_cfg: BacktestConfig,
    evaluation_start: pd.Timestamp,
) -> pd.DataFrame:
    """Evaluate nearby mappings without refitting or selecting a winner."""

    if not base_cfg.run_sensitivity:
        return pd.DataFrame()

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

    for profile_name, profile_cfg in sensitivity_profiles(base_cfg):
        raw_target, _ = build_alpha_targets(history, forecasts, profile_cfg)
        dynamic_target, _ = apply_state_machine(
            history,
            raw_target,
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
            pd.Series(
                average_exposure,
                index=evaluation_history.index,
            ),
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
            base_cfg.evaluation_fold_years,
        )
        valid = folds.dropna(
            subset=["return_alpha_dynamic_cagr"]
        )

        rows.append(
            {
                "profile": profile_name,
                "probability_offset": (
                    profile_cfg.probability_tier_1
                    - base_cfg.probability_tier_1
                ),
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
                "average_target_margin_pct": alpha_metrics[
                    "average_target_margin_pct"
                ],
                "annual_trades": alpha_metrics["annual_trades"],
                "fold_win_rate_vs_fixed_matched": (
                    float(valid["alpha_beats_fixed_matched"].mean())
                    if not valid.empty
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)


def run_full_validation(
    history: pd.DataFrame,
    prices: pd.Series,
    alpha_cfg: ReturnAlphaConfig,
    alpha_state_cfg: StateMachineConfig,
    current_state_cfg: StateMachineConfig,
    bt_cfg: BacktestConfig,
) -> Dict[str, Any]:
    common = history.index.intersection(prices.index)
    history = history.reindex(common)
    prices = prices.reindex(common)
    forecasts, model_records = generate_oos_forecasts(history, prices, alpha_cfg)
    alpha_raw, alpha_records = build_alpha_targets(history, forecasts, alpha_cfg)
    alpha_dynamic, state_records = apply_state_machine(history, alpha_raw, alpha_state_cfg)

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
        "forecast_diagnostics": forecast_diagnostics(prices.loc[evaluation_start:], alpha_dynamic.loc[evaluation_start:], 60),
    }
    folds = fold_analysis(ledgers, alpha_cfg.evaluation_fold_years)
    sensitivity = run_parameter_sensitivity(
        history,
        prices,
        forecasts,
        alpha_cfg,
        alpha_state_cfg,
        bt_cfg,
        evaluation_start,
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
    }


def validation_summary(result: Mapping[str, Any]) -> Dict[str, Any]:
    metrics_frame = result["metrics"].set_index("strategy")
    alpha = metrics_frame.loc["return_alpha_dynamic"]
    no_margin = metrics_frame.loc["no_margin"]
    matched = metrics_frame.loc["fixed_exposure_matched"]
    folds = result["folds"]
    valid_folds = folds.dropna(subset=["return_alpha_dynamic_cagr"])
    alpha_vs_matched = result["statistics"]["monthly_block_bootstrap_vs_fixed_matched"]
    sensitivity = result["sensitivity"]
    sensitivity_positive_rate = float(
        (sensitivity["alpha_minus_fixed_matched_cagr"] > 0).mean()
    ) if not sensitivity.empty else 0.0

    checks = {
        "alpha_cagr_above_no_margin": float(alpha["cagr"]) > float(no_margin["cagr"]),
        "alpha_cagr_above_fixed_exposure_matched": float(alpha["cagr"]) > float(matched["cagr"]),
        "alpha_drawdown_not_worse_than_no_margin_by_5pp": float(alpha["max_drawdown"]) >= float(no_margin["max_drawdown"]) - 0.05,
        "alpha_annual_trades_below_30": float(alpha["annual_trades"]) <= 30.0,
        "bootstrap_probability_positive_above_70pct": float(alpha_vs_matched.get("probability_positive") or 0.0) >= 0.70,
        "fold_win_rate_vs_fixed_matched_above_60pct": (
            float(valid_folds["alpha_beats_fixed_matched"].mean()) >= 0.60 if not valid_folds.empty else False
        ),
    }

    if not sensitivity.empty:
        checks["sensitivity_positive_rate_above_60pct"] = (
            sensitivity_positive_rate >= 0.60
        )
    return {
        "strict_pit_ready": bool(result["pit_ready"]),
        "checks": checks,
        "sensitivity_positive_rate": sensitivity_positive_rate,
        "passed_count": int(sum(checks.values())),
        "total_count": len(checks),
        "conclusion": (
            "PROMISING_BUT_REQUIRES_STRICT_PIT"
            if sum(checks.values()) >= 5 and not result["pit_ready"]
            else "PASS" if all(checks.values()) and result["pit_ready"]
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
    result = run_full_validation(
        history, prices, alpha_cfg, alpha_state_cfg, current_state_cfg, bt_cfg
    )
    write_validation_outputs(Path(args.output_dir), result)
    print(result["metrics"].to_string(index=False))
    print(json.dumps(validation_summary(result), ensure_ascii=False, indent=2))


def command_live(args: argparse.Namespace) -> None:
    history, prices = load_history_and_price(args)
    alpha_cfg = load_dataclass_config(args.alpha_config, ReturnAlphaConfig)
    alpha_cfg.ticker = args.ticker
    alpha_state_cfg = load_existing_config(args.alpha_state_config, StateMachineConfig)

    latest_forecast, model_records = generate_latest_forecast(
        history,
        prices,
        alpha_cfg,
    )
    latest_date = history.index[-1]
    latest_row = history.loc[latest_date]
    demand, alpha_reasons = alpha_demand_from_forecast(
        latest_forecast,
        alpha_cfg,
    )
    risk_cap, risk_reasons = risk_cap_from_scores(
        latest_row,
        alpha_cfg,
    )
    raw_target = min(demand, risk_cap)

    signal_payload = json.loads(Path(args.signals).read_text(encoding="utf-8"))
    signal = signal_from_history_row(latest_date, latest_row)
    signal["effective_date"] = signal_payload.get("decision_date", latest_date.date().isoformat())
    signal["Target_Margin_Pct"] = raw_target
    signal["Risk_Overrides"] = signal_payload.get("risk_overrides", [])
    signal["Critical_Data_Ok"] = signal_payload.get("data_quality", {}).get("critical_data_ok", True)

    state = load_margin_state(Path(args.state), args.initial_actual_margin_pct)
    new_state, decision = update_margin_state(signal, state, alpha_state_cfg)
    save_json(Path(args.state), asdict(new_state))

    reference_equity = float(args.reference_equity)
    execution = {
        "signal_date": latest_date.date().isoformat(),
        "effective_date": signal["effective_date"],
        "expected_net_return_60d_pct": 100.0 * float(latest_forecast["expected_net_return_60d"]),
        "probability_positive": float(latest_forecast["probability_positive"]),
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
        "recommended_loan_amount": reference_equity * float(decision["model_actual_margin_pct"]) / 100.0,
        "model_train_end": latest_forecast.get("model_train_end"),
        "strict_pit_ready": not history.get(
            "historical_data_is_revised",
            pd.Series(True, index=history.index),
        ).map(
            lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"}
        ).any(),
        "note": "Return-seeking forecast is experimental until strict PIT and live paper-trade validation pass.",
    }
    save_json(Path(args.execution_output), execution)
    signal_payload["return_alpha"] = execution
    save_json(Path(args.signals), signal_payload)
    print(json.dumps(json_safe(execution), ensure_ascii=False, indent=2))


def command_self_test(_: argparse.Namespace) -> None:
    rng = np.random.default_rng(415)
    dates = pd.bdate_range("2000-01-03", periods=1800)
    regime = np.sin(np.arange(len(dates)) / 90.0)
    returns = 0.00025 + 0.00045 * regime + rng.normal(0, 0.009, len(dates))
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
    history["Target_Margin_Pct"] = np.where(regime > 0, 5.0, 0.0)
    history["historical_data_is_revised"] = False

    cfg = ReturnAlphaConfig(
        horizons=(20, 40),
        horizon_weights=(0.5, 0.5),
        min_train_days=500,
        train_window_days=1000,
        bootstrap_iterations=100,
        run_sensitivity=False,
    )
    state_cfg = StateMachineConfig(
        increase_confirm_days=2,
        decrease_confirm_days=1,
        max_increase_step_pct=2.0,
        max_decrease_step_pct=4.0,
    )
    bt_cfg = BacktestConfig()
    result = run_full_validation(history, prices, cfg, state_cfg, StateMachineConfig(), bt_cfg)
    assert result["forecasts"]["expected_net_return_60d"].notna().sum() > 100
    assert result["alpha_dynamic"].max() <= cfg.max_alpha_margin_pct
    assert "fixed_exposure_matched" in result["ledgers"]

    # Look-ahead audit: changing the final tail must not alter forecasts well before the changed tail.
    altered = prices.copy()
    altered.iloc[-100:] *= np.linspace(1.0, 1.5, 100)
    forecast_a, _ = generate_oos_forecasts(history, prices, cfg)
    forecast_b, _ = generate_oos_forecasts(history, altered, cfg)
    safe_end = dates[-100 - max(cfg.horizons) - 5]
    comparison = pd.concat(
        [forecast_a.loc[:safe_end, "expected_net_return_60d"], forecast_b.loc[:safe_end, "expected_net_return_60d"]],
        axis=1,
    ).dropna()
    assert np.allclose(comparison.iloc[:, 0], comparison.iloc[:, 1], atol=1e-12)
    print("Return alpha self-test passed.")
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
