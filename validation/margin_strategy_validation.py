#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import yfinance as yf
except ModuleNotFoundError:
    yf = None


@dataclass
class StateMachineConfig:
    max_target_margin_pct: float = 20.0
    no_trade_band_pct: float = 1.0
    increase_trigger_pct: float = 1.5
    decrease_trigger_pct: float = 1.0
    increase_confirm_days: int = 3
    decrease_confirm_days: int = 2
    max_increase_step_pct: float = 1.0
    max_decrease_step_pct: float = 2.0
    min_business_days_between_trades: int = 2
    increase_cooldown_business_days: int = 5
    increase_once_per_iso_week: bool = True
    hard_exit_risk_score: float = 70.0
    hard_exit_liquidity_score: float = 30.0
    hard_exit_margin_score: float = 45.0
    hard_exit_coverage_ratio: float = 0.60
    critical_data_failure_is_hard_exit: bool = True
    risk_override_is_hard_exit: bool = True


@dataclass
class BacktestConfig:
    ticker: str = "QQQ"
    initial_equity: float = 1_000_000.0
    fixed_margin_pct: float = 5.0
    annual_borrow_rate: float = 0.04
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    trading_days_per_year: int = 252
    monthly_reference_equity: bool = True
    execution_lag_days: int = 1
    train_years: int = 5
    test_years: int = 2


@dataclass
class MarginState:
    model_actual_margin_pct: float = 0.0
    confirmed_target_margin_pct: float = 0.0
    increase_streak: int = 0
    decrease_streak: int = 0
    last_trade_date: Optional[str] = None
    last_increase_date: Optional[str] = None
    last_increase_iso_week: Optional[str] = None
    last_processed_signal_date: Optional[str] = None
    last_action: str = "INITIALIZE"
    last_reason: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.last_reason is None:
            self.last_reason = []


REQUIRED_HISTORY_COLUMNS = {
    "date", "Buy_Score", "Risk_Score", "Margin_Score", "Liquidity",
    "Target_Margin_Pct", "Coverage_Ratio",
}


def json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_config(path: Optional[str], cls: Any) -> Any:
    instance = cls()
    if not path:
        return instance
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {unknown}")
    values = asdict(instance)
    values.update(payload)
    return cls(**values)


def business_days_between(start: Optional[str], end: pd.Timestamp) -> int:
    if not start:
        return 10**9
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    if end_date <= start_date:
        return 0
    return int(np.busday_count(start_date, end_date))


def iso_week_key(date: pd.Timestamp) -> str:
    iso = date.isocalendar()
    return f"{int(iso.year)}-W{int(iso.week):02d}"


def signal_from_latest_json(payload: Mapping[str, Any]) -> Dict[str, Any]:
    quality = payload.get("data_quality", {})
    return {
        "signal_date": payload.get("signal_date"),
        "effective_date": payload.get("decision_date", payload.get("signal_date")),
        "Target_Margin_Pct": payload.get("allocation", {}).get("Target_Margin_Pct"),
        "Risk_Score": payload.get("scores", {}).get("Risk_Score"),
        "Margin_Score": payload.get("scores", {}).get("Margin_Score"),
        "Liquidity": payload.get("modules", {}).get("Liquidity"),
        "Coverage_Ratio": quality.get("coverage_ratio"),
        "Critical_Data_Ok": quality.get("critical_data_ok", True),
        "Risk_Overrides": payload.get("risk_overrides", []),
    }


def hard_exit_reasons(signal: Mapping[str, Any], cfg: StateMachineConfig) -> List[str]:
    reasons: List[str] = []
    risk = to_float(signal.get("Risk_Score"))
    liquidity = to_float(signal.get("Liquidity"))
    margin = to_float(signal.get("Margin_Score"))
    coverage = to_float(signal.get("Coverage_Ratio"))
    critical_ok = to_bool(signal.get("Critical_Data_Ok"), True)
    overrides = signal.get("Risk_Overrides", [])

    if risk is None or risk > cfg.hard_exit_risk_score:
        reasons.append("Risk_Score_Hard_Exit")
    if liquidity is None or liquidity < cfg.hard_exit_liquidity_score:
        reasons.append("Liquidity_Hard_Exit")
    if margin is None or margin < cfg.hard_exit_margin_score:
        reasons.append("Margin_Score_Hard_Exit")
    if coverage is None or coverage < cfg.hard_exit_coverage_ratio:
        reasons.append("Coverage_Hard_Exit")
    if cfg.critical_data_failure_is_hard_exit and not critical_ok:
        reasons.append("Critical_Data_Hard_Exit")

    if cfg.risk_override_is_hard_exit:
        if isinstance(overrides, str):
            active = [x.strip() for x in overrides.split(",") if x.strip()]
        elif isinstance(overrides, Iterable):
            active = list(overrides)
        else:
            active = []
        if active:
            reasons.append("Risk_Override_Hard_Exit")
    return reasons


def update_margin_state(
    signal: Mapping[str, Any],
    state: MarginState,
    cfg: StateMachineConfig,
) -> Tuple[MarginState, Dict[str, Any]]:
    s = copy.deepcopy(state)
    signal_date = pd.Timestamp(signal["signal_date"]).normalize()
    effective_date = pd.Timestamp(signal.get("effective_date", signal_date)).normalize()
    signal_date_str = signal_date.date().isoformat()
    raw = float(np.clip(to_float(signal.get("Target_Margin_Pct"), 0.0) or 0.0, 0, cfg.max_target_margin_pct))

    if s.last_processed_signal_date == signal_date_str:
        return s, {
            "signal_date": signal_date_str,
            "effective_date": effective_date.date().isoformat(),
            "raw_target_margin_pct": raw,
            "confirmed_target_margin_pct": s.confirmed_target_margin_pct,
            "model_actual_margin_pct": s.model_actual_margin_pct,
            "action": "NO_NEW_SIGNAL",
            "reasons": ["Signal_Date_Already_Processed"],
            "hard_exit": False,
        }

    current = float(s.model_actual_margin_pct)
    reasons = hard_exit_reasons(signal, cfg)
    hard_exit = bool(reasons)
    action = "HOLD"

    if hard_exit:
        s.confirmed_target_margin_pct = 0.0
        s.model_actual_margin_pct = 0.0
        s.increase_streak = 0
        s.decrease_streak = 0
        action = "EXIT_ALL" if current > 0 else "HOLD_ZERO"
        if current > 0:
            s.last_trade_date = signal_date_str
    else:
        difference = raw - current
        if abs(difference) < cfg.no_trade_band_pct:
            s.increase_streak = 0
            s.decrease_streak = 0
            reasons = ["Inside_No_Trade_Band"]
        elif difference >= cfg.increase_trigger_pct:
            s.increase_streak += 1
            s.decrease_streak = 0
            reasons = [f"Increase_Confirmation_{s.increase_streak}_of_{cfg.increase_confirm_days}"]
            weekly_ok = (not cfg.increase_once_per_iso_week or s.last_increase_iso_week != iso_week_key(signal_date))
            eligible = (
                s.increase_streak >= cfg.increase_confirm_days
                and business_days_between(s.last_trade_date, signal_date) >= cfg.min_business_days_between_trades
                and business_days_between(s.last_increase_date, signal_date) >= cfg.increase_cooldown_business_days
                and weekly_ok
            )
            if eligible:
                new_target = min(raw, current + cfg.max_increase_step_pct, cfg.max_target_margin_pct)
                s.confirmed_target_margin_pct = new_target
                s.model_actual_margin_pct = new_target
                s.last_trade_date = signal_date_str
                s.last_increase_date = signal_date_str
                s.last_increase_iso_week = iso_week_key(signal_date)
                s.increase_streak = 0
                action = "INCREASE_MARGIN"
                reasons.append("Increase_Executed")
            else:
                action = "WAIT_INCREASE_CONFIRMATION"
        elif difference <= -cfg.decrease_trigger_pct:
            s.decrease_streak += 1
            s.increase_streak = 0
            reasons = [f"Decrease_Confirmation_{s.decrease_streak}_of_{cfg.decrease_confirm_days}"]
            eligible = (
                s.decrease_streak >= cfg.decrease_confirm_days
                and business_days_between(s.last_trade_date, signal_date) >= cfg.min_business_days_between_trades
            )
            if eligible:
                new_target = max(raw, current - cfg.max_decrease_step_pct, 0.0)
                s.confirmed_target_margin_pct = new_target
                s.model_actual_margin_pct = new_target
                s.last_trade_date = signal_date_str
                s.decrease_streak = 0
                action = "REPAY_ALL" if new_target <= 0 else "DECREASE_MARGIN"
                reasons.append("Decrease_Executed")
            else:
                action = "WAIT_DECREASE_CONFIRMATION"
        else:
            s.increase_streak = 0
            s.decrease_streak = 0
            reasons = ["No_Trigger"]

    s.last_processed_signal_date = signal_date_str
    s.last_action = action
    s.last_reason = reasons
    decision = {
        "signal_date": signal_date_str,
        "effective_date": effective_date.date().isoformat(),
        "raw_target_margin_pct": round(raw, 4),
        "confirmed_target_margin_pct": round(s.confirmed_target_margin_pct, 4),
        "model_actual_margin_pct": round(s.model_actual_margin_pct, 4),
        "action": action,
        "reasons": reasons,
        "hard_exit": hard_exit,
        "increase_streak": s.increase_streak,
        "decrease_streak": s.decrease_streak,
    }
    return s, decision


def load_score_history(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_HISTORY_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"score_history.csv missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).drop_duplicates("date", keep="last").set_index("date").sort_index()
    for col in ["Buy_Score", "Risk_Score", "Margin_Score", "Liquidity", "Target_Margin_Pct", "Coverage_Ratio"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "Critical_Data_Ok" not in frame.columns:
        frame["Critical_Data_Ok"] = True
    else:
        frame["Critical_Data_Ok"] = frame["Critical_Data_Ok"].map(lambda x: to_bool(x, True))
    return frame


def history_pit_status(history: pd.DataFrame) -> Tuple[bool, str]:
    if "historical_data_is_revised" not in history.columns:
        return False, "PIT status unknown: historical_data_is_revised is absent."
    revised = history["historical_data_is_revised"].map(to_bool).any()
    if revised:
        return False, "Exploratory only: live_public history may contain revised data."
    return True, "History is marked unrevised; upstream release timestamps must still be audited."


def signal_from_history_row(date: pd.Timestamp, row: pd.Series) -> Dict[str, Any]:
    return {
        "signal_date": date.date().isoformat(),
        "effective_date": date.date().isoformat(),
        "Target_Margin_Pct": row.get("Target_Margin_Pct"),
        "Risk_Score": row.get("Risk_Score"),
        "Margin_Score": row.get("Margin_Score"),
        "Liquidity": row.get("Liquidity"),
        "Coverage_Ratio": row.get("Coverage_Ratio"),
        "Critical_Data_Ok": row.get("Critical_Data_Ok", True),
        "Risk_Overrides": row.get("Risk_Overrides", []),
    }


def generate_dynamic_targets(history: pd.DataFrame, cfg: StateMachineConfig) -> Tuple[pd.Series, pd.DataFrame]:
    state = MarginState()
    targets: Dict[pd.Timestamp, float] = {}
    rows: List[Dict[str, Any]] = []
    for date, row in history.iterrows():
        state, decision = update_margin_state(signal_from_history_row(date, row), state, cfg)
        targets[date] = state.model_actual_margin_pct
        rows.append({**decision, "Risk_Score": row.get("Risk_Score"), "Margin_Score": row.get("Margin_Score"), "Liquidity": row.get("Liquidity"), "Coverage_Ratio": row.get("Coverage_Ratio")})
    return pd.Series(targets, name="Dynamic_Target_Margin_Pct"), pd.DataFrame(rows)


def load_prices(ticker: str, start: pd.Timestamp, end: pd.Timestamp, price_csv: Optional[Path]) -> pd.Series:
    if price_csv:
        frame = pd.read_csv(price_csv)
        date_col = next((c for c in ["date", "Date"] if c in frame.columns), None)
        price_col = next((c for c in ["Adj Close", "adj_close", "Close", "close"] if c in frame.columns), None)
        if not date_col or not price_col:
            raise ValueError("Price CSV needs date/Date and Adj Close/Close.")
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame[price_col] = pd.to_numeric(frame[price_col], errors="coerce")
        series = frame.dropna(subset=[date_col, price_col]).drop_duplicates(date_col, keep="last").set_index(date_col)[price_col].sort_index()
    else:
        if yf is None:
            raise RuntimeError("yfinance is required when --price-csv is not provided.")
        raw = yf.download(ticker, start=(start - pd.Timedelta(days=10)).date(), end=(end + pd.Timedelta(days=10)).date(), auto_adjust=True, progress=False, threads=False)
        if raw.empty:
            raise RuntimeError(f"No prices returned for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            close_cols = [c for c in raw.columns if c[0] == "Close"]
            if not close_cols:
                raise RuntimeError("No Close column in yfinance data.")
            series = raw[close_cols[0]]
        else:
            series = raw["Close"]
    series.index = pd.to_datetime(series.index).tz_localize(None)
    return pd.to_numeric(series, errors="coerce").dropna().sort_index().loc[start:end]


def align_history_prices(history: pd.DataFrame, prices: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    common = history.index.intersection(prices.index)
    if len(common) < 60:
        raise ValueError("Fewer than 60 common trading days.")
    return history.loc[common], prices.loc[common]


def simulate_account(prices: pd.Series, target_pct: pd.Series, cfg: BacktestConfig, name: str) -> pd.DataFrame:
    target_pct = target_pct.reindex(prices.index).ffill().fillna(0.0)
    effective = target_pct.shift(cfg.execution_lag_days).fillna(0.0)
    returns = prices.pct_change().fillna(0.0)
    equity = cfg.initial_equity
    loan = 0.0
    assets = equity
    reference_equity = equity
    previous_month: Optional[Tuple[int, int]] = None
    total_cost_rate = (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000.0
    rows: List[Dict[str, Any]] = []

    for date in prices.index:
        month_key = (date.year, date.month)
        if not cfg.monthly_reference_equity or previous_month is None or month_key != previous_month:
            reference_equity = max(equity, 0.0)
            previous_month = month_key

        target = float(np.clip(effective.loc[date], 0, 100))
        target_loan = reference_equity * target / 100.0
        delta = target_loan - loan
        equity_before = equity
        assets += delta
        loan = target_loan
        transaction_cost = abs(delta) * total_cost_rate
        assets -= transaction_cost
        assets *= 1.0 + float(returns.loc[date])
        interest_cost = loan * cfg.annual_borrow_rate / cfg.trading_days_per_year
        assets -= interest_cost
        equity = max(assets - loan, 0.0)
        strategy_return = equity / equity_before - 1.0 if equity_before > 0 else -1.0
        rows.append({
            "date": date, "strategy": name, "price": float(prices.loc[date]),
            "underlying_return": float(returns.loc[date]), "signal_target_margin_pct": float(target_pct.loc[date]),
            "effective_target_margin_pct": target, "reference_equity": reference_equity,
            "loan_change": delta, "loan_balance": loan, "asset_value": assets, "equity": equity,
            "strategy_return": strategy_return, "actual_margin_pct": 100 * loan / equity if loan > 0 and equity > 0 else 0.0,
            "transaction_cost": transaction_cost, "interest_cost": interest_cost,
            "maintenance_ratio": 100 * assets / loan if loan > 0 else np.inf,
            "trade_direction": int(np.sign(delta)),
        })
        if equity <= 0:
            break
    return pd.DataFrame(rows).set_index("date")


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def count_whipsaws(ledger: pd.DataFrame, window_days: int = 5) -> int:
    trades = ledger.loc[ledger["trade_direction"] != 0, ["trade_direction"]]
    count = 0
    for i in range(1, len(trades)):
        if trades.iloc[i, 0] == trades.iloc[i - 1, 0]:
            continue
        if np.busday_count(trades.index[i - 1].date(), trades.index[i].date()) <= window_days:
            count += 1
    return count


def metrics(ledger: pd.DataFrame, trading_days: int = 252) -> Dict[str, Any]:
    r = ledger["strategy_return"].fillna(0.0)
    e = ledger["equity"]
    years = max((ledger.index[-1] - ledger.index[0]).days / 365.25, len(ledger) / trading_days, 1 / trading_days)
    total = e.iloc[-1] / e.iloc[0] - 1.0
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1.0 if e.iloc[-1] > 0 else -1.0
    vol = float(r.std(ddof=0) * math.sqrt(trading_days))
    sharpe = float(r.mean() * trading_days / vol) if vol > 0 else None
    downside = float(r.clip(upper=0).std(ddof=0) * math.sqrt(trading_days))
    sortino = float(r.mean() * trading_days / downside) if downside > 0 else None
    dd = max_drawdown(e)
    monthly = (1 + r).resample("ME").prod() - 1
    trades = int((ledger["loan_change"].abs() > 1e-9).sum())
    turnover = float(ledger["loan_change"].abs().sum() / max(e.mean(), 1e-12) / years)
    return json_safe({
        "strategy": ledger["strategy"].iloc[0], "start_date": ledger.index[0].date().isoformat(),
        "end_date": ledger.index[-1].date().isoformat(), "trading_days": len(ledger), "total_return": total,
        "cagr": cagr, "annual_volatility": vol, "sharpe": sharpe, "sortino": sortino,
        "max_drawdown": dd, "calmar": cagr / abs(dd) if dd < 0 else None,
        "positive_day_rate": float((r > 0).mean()), "positive_month_rate": float((monthly > 0).mean()) if len(monthly) else None,
        "worst_day": float(r.min()), "trade_count": trades, "annual_trades": trades / years,
        "annual_turnover": turnover, "whipsaw_count_5d": count_whipsaws(ledger),
        "average_target_margin_pct": float(ledger["effective_target_margin_pct"].mean()),
        "average_actual_margin_pct": float(ledger["actual_margin_pct"].replace([np.inf, -np.inf], np.nan).mean()),
        "max_actual_margin_pct": float(ledger["actual_margin_pct"].replace([np.inf, -np.inf], np.nan).max()),
        "minimum_maintenance_ratio": float(ledger["maintenance_ratio"].replace([np.inf, -np.inf], np.nan).min()) if ledger["loan_balance"].gt(0).any() else None,
        "interest_cost_pct_initial_equity": float(ledger["interest_cost"].sum() / ledger["equity"].iloc[0]),
        "transaction_cost_pct_initial_equity": float(ledger["transaction_cost"].sum() / ledger["equity"].iloc[0]),
        "final_equity": float(e.iloc[-1]),
    })


def run_strategy_set(history: pd.DataFrame, prices: pd.Series, state_cfg: StateMachineConfig, bt_cfg: BacktestConfig):
    dynamic, decisions = generate_dynamic_targets(history, state_cfg)
    targets = {
        "no_margin": pd.Series(0.0, index=history.index),
        "fixed_margin": pd.Series(bt_cfg.fixed_margin_pct, index=history.index),
        "raw_daily_margin": history["Target_Margin_Pct"].fillna(0.0),
        "dynamic_state_machine": dynamic,
    }
    ledgers = {name: simulate_account(prices, target, bt_cfg, name) for name, target in targets.items()}
    summary = pd.DataFrame([metrics(ledger, bt_cfg.trading_days_per_year) for ledger in ledgers.values()])
    return ledgers, summary, decisions


def parameter_grid(base: StateMachineConfig) -> List[StateMachineConfig]:
    grid = {
        "no_trade_band_pct": [0.75, 1.0, 1.25],
        "increase_confirm_days": [2, 3, 5],
        "decrease_confirm_days": [1, 2],
        "max_increase_step_pct": [0.5, 1.0],
        "max_decrease_step_pct": [1.0, 2.0],
    }
    result = []
    for values in itertools.product(*grid.values()):
        payload = asdict(base)
        payload.update(dict(zip(grid.keys(), values)))
        result.append(StateMachineConfig(**payload))
    return result


def objective(candidate: Mapping[str, Any], benchmark: Mapping[str, Any]) -> float:
    cagr, base_cagr = float(candidate.get("cagr") or -1), float(benchmark.get("cagr") or -1)
    dd, base_dd = float(candidate.get("max_drawdown") or -1), float(benchmark.get("max_drawdown") or -1)
    trades = float(candidate.get("annual_trades") or 0)
    if dd < base_dd - 0.03 or trades > 30:
        return -1e9
    return 100 * (cagr - base_cagr) + 0.5 * float(candidate.get("calmar") or -10) + 0.25 * float(candidate.get("sharpe") or -10) - 0.03 * trades - 0.1 * float(candidate.get("annual_turnover") or 0)


def walk_forward(history: pd.DataFrame, prices: pd.Series, state_cfg: StateMachineConfig, bt_cfg: BacktestConfig):
    years = sorted(set(history.index.year))
    if len(years) < bt_cfg.train_years + bt_cfg.test_years:
        raise ValueError("Not enough years for walk-forward.")
    rows = []
    pos = bt_cfg.train_years
    while pos < len(years):
        train_years = years[:pos]
        test_years = years[pos:pos + bt_cfg.test_years]
        if not test_years:
            break
        train = history.loc[f"{train_years[0]}-01-01":f"{train_years[-1]}-12-31"]
        train_prices = prices.reindex(train.index).dropna(); train = train.reindex(train_prices.index)
        base_ledger = simulate_account(train_prices, pd.Series(0.0, index=train.index), bt_cfg, "no_margin")
        base_m = metrics(base_ledger, bt_cfg.trading_days_per_year)
        best_cfg, best_score = None, -np.inf
        for cfg in parameter_grid(state_cfg):
            target, _ = generate_dynamic_targets(train, cfg)
            ledger = simulate_account(train_prices, target, bt_cfg, "dynamic_state_machine")
            score = objective(metrics(ledger, bt_cfg.trading_days_per_year), base_m)
            if score > best_score:
                best_score, best_cfg = score, cfg
        if best_cfg is None:
            raise RuntimeError("No valid walk-forward parameters.")
        test = history.loc[f"{test_years[0]}-01-01":f"{test_years[-1]}-12-31"]
        test_prices = prices.reindex(test.index).dropna(); test = test.reindex(test_prices.index)
        warmed_target, _ = generate_dynamic_targets(history.loc[:test.index.max()], best_cfg)
        strategies = {
            "no_margin": pd.Series(0.0, index=test.index),
            "fixed_margin": pd.Series(bt_cfg.fixed_margin_pct, index=test.index),
            "raw_daily_margin": test["Target_Margin_Pct"].fillna(0.0),
            "dynamic_state_machine": warmed_target.reindex(test.index).ffill().fillna(0.0),
        }
        fold_metrics = {name: metrics(simulate_account(test_prices, target, bt_cfg, name), bt_cfg.trading_days_per_year) for name, target in strategies.items()}
        rows.append({
            "train_start": train.index.min().date().isoformat(), "train_end": train.index.max().date().isoformat(),
            "test_start": test.index.min().date().isoformat(), "test_end": test.index.max().date().isoformat(),
            "selected_parameters": json.dumps(asdict(best_cfg), ensure_ascii=False, sort_keys=True),
            "training_objective": best_score,
            "test_no_margin_cagr": fold_metrics["no_margin"]["cagr"],
            "test_fixed_margin_cagr": fold_metrics["fixed_margin"]["cagr"],
            "test_raw_daily_cagr": fold_metrics["raw_daily_margin"]["cagr"],
            "test_dynamic_cagr": fold_metrics["dynamic_state_machine"]["cagr"],
            "test_dynamic_max_drawdown": fold_metrics["dynamic_state_machine"]["max_drawdown"],
            "test_dynamic_annual_trades": fold_metrics["dynamic_state_machine"]["annual_trades"],
        })
        pos += bt_cfg.test_years
    return pd.DataFrame(rows)


def command_live(args: argparse.Namespace) -> None:
    payload = json.loads(Path(args.signals).read_text(encoding="utf-8"))
    state_path = Path(args.state)
    state = MarginState(**json.loads(state_path.read_text(encoding="utf-8"))) if state_path.exists() else MarginState(model_actual_margin_pct=args.initial_actual_margin_pct, confirmed_target_margin_pct=args.initial_actual_margin_pct)
    cfg = load_config(args.state_config, StateMachineConfig)
    state, decision = update_margin_state(signal_from_latest_json(payload), state, cfg)
    save_json(state_path, asdict(state))
    reference = args.reference_equity
    execution = {**decision, "reference_equity": reference, "recommended_loan_amount": reference * decision["model_actual_margin_pct"] / 100 if reference else None, "note": "model_actual_margin_pct is a persistent model target, not broker-confirmed debt."}
    save_json(Path(args.execution_output), execution)
    if args.patch_signals:
        payload["margin_execution"] = execution
        save_json(Path(args.signals), payload)
    print(json.dumps(execution, ensure_ascii=False, indent=2))


def command_backtest(args: argparse.Namespace) -> None:
    history = load_score_history(Path(args.history)); pit_ok, pit_message = history_pit_status(history)
    if args.strict_pit and not pit_ok:
        raise RuntimeError(pit_message)
    prices = load_prices(args.ticker, history.index.min(), history.index.max(), Path(args.price_csv) if args.price_csv else None)
    history, prices = align_history_prices(history, prices)
    state_cfg = load_config(args.state_config, StateMachineConfig)
    bt_cfg = load_config(args.backtest_config, BacktestConfig); bt_cfg.ticker = args.ticker
    ledgers, summary, decisions = run_strategy_set(history, prices, state_cfg, bt_cfg)
    stress_cfg = copy.deepcopy(bt_cfg); stress_cfg.annual_borrow_rate += 0.02; stress_cfg.transaction_cost_bps *= 2; stress_cfg.slippage_bps *= 2
    _, stress_summary, _ = run_strategy_set(history, prices, state_cfg, stress_cfg)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "strategy_comparison.csv", index=False)
    stress_summary.to_csv(out / "strategy_comparison_stress.csv", index=False)
    decisions.to_csv(out / "state_machine_decisions.csv", index=False)
    for name, ledger in ledgers.items():
        ledger.to_csv(out / f"ledger_{name}.csv")
    indexed = summary.set_index("strategy"); base = indexed.loc["no_margin"]; dynamic = indexed.loc["dynamic_state_machine"]; raw = indexed.loc["raw_daily_margin"]
    checks = {
        "dynamic_cagr_above_no_margin": float(dynamic["cagr"]) > float(base["cagr"]),
        "dynamic_calmar_above_no_margin": float(dynamic["calmar"] or -np.inf) > float(base["calmar"] or -np.inf),
        "dynamic_drawdown_not_worse_than_3pp": float(dynamic["max_drawdown"]) >= float(base["max_drawdown"]) - 0.03,
        "dynamic_annual_trades_below_30": float(dynamic["annual_trades"]) <= 30,
        "dynamic_trades_less_than_raw_daily": float(dynamic["trade_count"]) < float(raw["trade_count"]),
    }
    save_json(out / "validation_report.json", {"pit_status": {"strict_pit_ready": pit_ok, "message": pit_message}, "checks": checks, "strategy_metrics": summary.to_dict("records"), "stress_metrics": stress_summary.to_dict("records")})
    print(summary.to_string(index=False)); print("\n", json.dumps(checks, ensure_ascii=False, indent=2))


def command_walk_forward(args: argparse.Namespace) -> None:
    history = load_score_history(Path(args.history)); pit_ok, pit_message = history_pit_status(history)
    if args.strict_pit and not pit_ok:
        raise RuntimeError(pit_message)
    prices = load_prices(args.ticker, history.index.min(), history.index.max(), Path(args.price_csv) if args.price_csv else None)
    history, prices = align_history_prices(history, prices)
    state_cfg = load_config(args.state_config, StateMachineConfig)
    bt_cfg = load_config(args.backtest_config, BacktestConfig); bt_cfg.ticker = args.ticker
    folds = walk_forward(history, prices, state_cfg, bt_cfg)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out / "walk_forward_folds.csv", index=False)
    summary = {
        "pit_status": {"strict_pit_ready": pit_ok, "message": pit_message},
        "fold_count": len(folds),
        "dynamic_oos_win_rate_vs_no_margin": float((folds["test_dynamic_cagr"] > folds["test_no_margin_cagr"]).mean()) if len(folds) else None,
        "dynamic_oos_win_rate_vs_fixed_margin": float((folds["test_dynamic_cagr"] > folds["test_fixed_margin_cagr"]).mean()) if len(folds) else None,
        "folds": folds.to_dict("records"),
    }
    save_json(out / "walk_forward_report.json", summary)
    print(folds.to_string(index=False)); print("\n", json.dumps(summary, ensure_ascii=False, indent=2))


def command_self_test(_: argparse.Namespace) -> None:
    dates = pd.bdate_range("2020-01-01", periods=600)
    regime = np.where(np.arange(600) < 300, 3.0, 4.0)
    noise = np.where(np.arange(600) % 2 == 0, 1.2, -1.2)
    raw = pd.Series(np.clip(regime + noise, 0.0, 8.0), index=dates)
    history = pd.DataFrame({"Buy_Score": 65.0, "Risk_Score": 35.0, "Margin_Score": 55.0, "Liquidity": 55.0, "Target_Margin_Pct": raw, "Coverage_Ratio": 1.0, "Critical_Data_Ok": True, "historical_data_is_revised": False}, index=dates)
    history.loc[dates[360:365], "Risk_Score"] = 80.0
    prices = pd.Series(100 * np.cumprod(1 + 0.0003 + 0.0001 * np.sin(np.arange(600) / 20)), index=dates)
    ledgers, summary, decisions = run_strategy_set(history, prices, StateMachineConfig(), BacktestConfig())
    assert decisions["action"].isin(["EXIT_ALL", "HOLD_ZERO"]).any()
    indexed = summary.set_index("strategy")
    assert indexed.loc["dynamic_state_machine", "trade_count"] < indexed.loc["raw_daily_margin", "trade_count"]
    assert all(ledger["equity"].iloc[-1] > 0 for ledger in ledgers.values())
    print("Self-test passed.\n", summary.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Investment OS persistent margin state and validation.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("live"); p.add_argument("--signals", default="output/latest_signals.json"); p.add_argument("--state", default="output/margin_state.json"); p.add_argument("--execution-output", default="output/margin_execution.json"); p.add_argument("--state-config"); p.add_argument("--initial-actual-margin-pct", type=float, default=0.0); p.add_argument("--reference-equity", type=float); p.add_argument("--patch-signals", action=argparse.BooleanOptionalAction, default=True); p.set_defaults(func=command_live)
    for name, func, default_out in [("backtest", command_backtest, "output/margin_validation"), ("walk-forward", command_walk_forward, "output/margin_walk_forward")]:
        p = sub.add_parser(name); p.add_argument("--history", default="output/score_history.csv"); p.add_argument("--ticker", default="QQQ"); p.add_argument("--price-csv"); p.add_argument("--state-config"); p.add_argument("--backtest-config"); p.add_argument("--output-dir", default=default_out); p.add_argument("--strict-pit", action="store_true"); p.set_defaults(func=func)
    p = sub.add_parser("self-test"); p.set_defaults(func=command_self_test)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
