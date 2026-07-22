from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# 1. Configuration
# ============================================================

@dataclass
class Config:
    data_mode: str = field(default_factory=lambda: os.getenv("INVESTMENT_OS_DATA_MODE", "live_public"))
    start_date: str = "2012-01-01"
    annual_borrow_rate: float = 0.04
    rolling_percentile_years: int = 10
    min_percentile_years: int = 3
    max_target_margin_pct: float = 20.0
    max_margin_without_valuation_or_carry: float = 5.0
    minimum_coverage_for_margin: float = 0.60
    data_directory: Path = Path("data")
    output_directory: Path = Path("output")
    cache_directory: Path = Path("data/cache")
    fred_api_key: Optional[str] = field(default_factory=lambda: os.getenv("FRED_API_KEY"))

    max_staleness_days: Dict[str, int] = field(default_factory=lambda: {
        "market_price": 5,
        "daily_market_indicator": 7,
        "weekly_macro": 21,
        "monthly_macro": 75,
        "quarterly_fundamental": 150,
    })

    module_base_weights: Dict[str, float] = field(default_factory=lambda: {
        "market_regime": 0.21, "ai_cycle": 0.14, "valuation": 0.12,
        "macro": 0.17, "liquidity": 0.19, "positioning": 0.17,
    })
    buy_score_weights: Dict[str, float] = field(default_factory=lambda: {
        "market_regime": 0.25, "ai_cycle": 0.15, "valuation": 0.20,
        "macro": 0.10, "liquidity": 0.10, "positioning": 0.20,
    })
    risk_support_weights: Dict[str, float] = field(default_factory=lambda: {
        "market_regime": 0.30, "ai_cycle": 0.10, "valuation": 0.05,
        "macro": 0.20, "liquidity": 0.25, "positioning": 0.10,
    })
    margin_base_weights: Dict[str, float] = field(default_factory=lambda: {
        "market_regime": 0.25, "ai_cycle": 0.05, "valuation": 0.20,
        "macro": 0.10, "liquidity": 0.25, "positioning": 0.15,
    })
    market_regime_weights: Dict[str, float] = field(default_factory=lambda: {
        "spy_trend": 0.25, "qqq_trend": 0.25, "soxx_trend": 0.20,
        "breadth": 0.15, "credit": 0.15,
    })
    macro_weights: Dict[str, float] = field(default_factory=lambda: {
        "core_pce": 0.25, "employment_gap": 0.20, "initial_claims": 0.25,
        "yield_curve": 0.20, "nowcast_or_pmi": 0.10,
    })
    liquidity_weights: Dict[str, float] = field(default_factory=lambda: {
        "nfl_13w_change": 0.35, "reserve_balances": 0.25,
        "rrp_inverse": 0.15, "tga_inverse": 0.15, "real_m2": 0.10,
    })
    positioning_weights: Dict[str, float] = field(default_factory=lambda: {
        "vix_percentile": 0.25, "vix_term_structure": 0.25,
        "skew": 0.15, "vvix": 0.15, "put_call": 0.10,
        "cftc_or_flow": 0.10,
    })

    market_tickers: Tuple[str, ...] = (
        "SPY", "QQQ", "SOXX", "IWM", "MDY", "SMH", "HYG", "NVDA",
        "^VIX", "^VIX3M", "^SKEW", "^VVIX", "^TNX",
    )

    # live_public uses conservative publication-lag proxies. strict_pit uses
    # effective_trade_date from data/macro_vintages.csv instead.
    fred_metadata: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "BAMLH0A0HYM2": {"alias": "hy_oas", "frequency": "daily", "lag_days": 1, "age": "daily_market_indicator", "unit": None},
        "DFII10": {"alias": "real_10y", "frequency": "daily", "lag_days": 1, "age": "daily_market_indicator", "unit": None},
        "T10Y3M": {"alias": "t10y3m", "frequency": "daily", "lag_days": 1, "age": "daily_market_indicator", "unit": None},
        "T10Y2Y": {"alias": "t10y2y", "frequency": "daily", "lag_days": 1, "age": "daily_market_indicator", "unit": None},
        "WALCL": {"alias": "walcl_bn", "frequency": "weekly", "lag_days": 2, "age": "weekly_macro", "unit": 0.001},
        "RRPONTSYD": {"alias": "rrp_bn", "frequency": "daily", "lag_days": 1, "age": "daily_market_indicator", "unit": 1.0},
        "WTREGEN": {"alias": "tga_bn", "frequency": "weekly", "lag_days": 2, "age": "weekly_macro", "unit": 0.001},
        "WRESBAL": {"alias": "reserves_bn", "frequency": "weekly", "lag_days": 2, "age": "weekly_macro", "unit": 0.001},
        "M2REAL": {"alias": "real_m2_bn", "frequency": "monthly", "lag_days": 35, "age": "monthly_macro", "unit": 1.0},
        "ICSA": {"alias": "claims", "frequency": "weekly", "lag_days": 1, "age": "weekly_macro", "unit": None},
        "UNRATE": {"alias": "unrate", "frequency": "monthly", "lag_days": 35, "age": "monthly_macro", "unit": None},
        "PCEPILFE": {"alias": "core_pce_index", "frequency": "monthly", "lag_days": 35, "age": "monthly_macro", "unit": None},
    })


CFG = Config()


def normalize_missing_labels(
    items: Iterable[str],
) -> List[str]:
    """
    Remove duplicate diagnostic labels such as:

    - cftc_or_flow + cftc_or_flow_score -> cftc_or_flow
    - tsmc_hpc_growth + tsmc_hpc_growth_score -> tsmc_hpc_growth
    - micron_dc_hbm_score_score -> micron_dc_hbm_score
    """

    cleaned: set[str] = set()

    for raw_item in items:
        label = str(
            raw_item
        ).strip()

        if not label:
            continue

        while label.endswith(
            "_score_score"
        ):
            label = label[:-6]

        cleaned.add(
            label
        )

    for label in list(
        cleaned
    ):
        if not label.endswith(
            "_score"
        ):
            continue

        base_label = label[:-6]

        if base_label in cleaned:
            cleaned.discard(
                label
            )

    return sorted(
        cleaned
    )


@dataclass
class AlignedSeries:
    value: pd.Series
    update_date: pd.Series
    age_days: pd.Series
    stale: pd.Series
    source: str
    is_proxy: bool = False


@dataclass
class ModuleResult:
    name: str
    score: pd.Series
    coverage: pd.Series
    components: pd.DataFrame
    weights: Dict[str, float]
    is_proxy: bool
    missing_inputs: List[str]
    stale_inputs: List[str]
    raw: Dict[str, pd.Series]

    def latest_quality(self, date: pd.Timestamp) -> Dict[str, Any]:
        return {
            "score": number(self.score.get(date), 2),
            "available": number(self.score.get(date)) is not None,
            "coverage": number(self.coverage.get(date), 4),
            "is_proxy": self.is_proxy,
            "effective_weights": effective_weights(self.components.loc[:date].tail(1), self.weights),
            "missing_inputs": normalize_missing_labels(self.missing_inputs),
            "stale_inputs": sorted(set(self.stale_inputs)),
            "last_updated": date.date().isoformat(),
        }


# ============================================================
# 2. Logging and common helpers
# ============================================================

def configure_logging(cfg: Config) -> logging.Logger:
    cfg.output_directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("investment_os")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(cfg.output_directory / "warnings.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def requests_session() -> requests.Session:
    """
    建立具有重試機制的 HTTP Session。

    不設定過多重試，避免 FRED 故障時，
    每一個資料序列都等待數分鐘。
    """
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=2.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )

    session = requests.Session()

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=10,
        pool_maxsize=10,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": "Investment-OS/1.0",
        "Accept": "application/json,text/csv,*/*",
    })

    return session


def number(value: Any, digits: int = 4) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def as_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "y"}


def validate_config(cfg: Config) -> None:
    if cfg.data_mode not in {"live_public", "strict_pit", "market_only"}:
        raise ValueError("data_mode must be live_public, strict_pit, or market_only")
    if not 0 <= cfg.annual_borrow_rate <= 0.30:
        raise ValueError("annual_borrow_rate must be between 0 and 0.30")
    if not 0 < cfg.max_target_margin_pct <= 20:
        raise ValueError("max_target_margin_pct must be in (0, 20]")

def next_nyse_session_after(
    signal_date: pd.Timestamp,
) -> pd.Timestamp:
    """
    取得 signal_date 之後的下一個 NYSE 交易日。

    signal_date：
        最新完成的市場收盤日

    回傳值：
        這份訊號真正用來做交易決策的日期
    """

    signal = pd.Timestamp(
        signal_date
    ).tz_localize(None).normalize()

    nyse = mcal.get_calendar("NYSE")

    schedule = nyse.schedule(
        start_date=(
            signal
            + pd.Timedelta(days=1)
        ).date(),
        end_date=(
            signal
            + pd.Timedelta(days=14)
        ).date(),
    )

    if schedule.empty:
        raise RuntimeError(
            "Unable to resolve next NYSE session."
        )

    return pd.Timestamp(
        schedule.index[0]
    ).tz_localize(None)


# ============================================================
# 3. Point-in-time-safe statistics
# ============================================================

def periods_per_year(freq: str) -> int:
    return {"daily": 252, "weekly": 52, "monthly": 12, "quarterly": 4}[freq]


def winsorize_series(series: pd.Series, frequency: str = "daily", years: int = 10) -> pd.Series:
    """Rolling 1%-99% winsorization using no future values."""
    s = pd.to_numeric(series, errors="coerce").sort_index()
    ppy = periods_per_year(frequency)
    window = years * ppy
    minp = min(window, max(10, ppy))
    lo = s.rolling(window, min_periods=minp).quantile(0.01)
    hi = s.rolling(window, min_periods=minp).quantile(0.99)
    out = s.copy()
    mask = lo.notna() & hi.notna()
    out.loc[mask] = s.loc[mask].clip(lo.loc[mask], hi.loc[mask])
    return out


def ewma_smooth(series: pd.Series, span: int = 5) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=span, adjust=False, min_periods=1).mean()


def rolling_percentile(series: pd.Series, frequency: str,
                       years: int = CFG.rolling_percentile_years,
                       min_years: int = CFG.min_percentile_years) -> pd.Series:
    """Trailing percentile; each row only uses data available up to that row."""
    s = pd.to_numeric(series, errors="coerce").sort_index()
    ppy = periods_per_year(frequency)
    window, minp = years * ppy, min_years * ppy

    def last_pct(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return np.nan
        x = values[-1]
        return 100.0 * (np.sum(values < x) + 0.5 * np.sum(values == x)) / len(values)

    return s.rolling(window, min_periods=minp).apply(last_pct, raw=True)



def rolling_percentile_with_fallback(
    series: pd.Series,
    frequency: str,
    fallback_years: int = 1,
) -> pd.Series:
    """
    Use the configured trailing percentile when sufficient history
    exists. If a cache or newly added series has less than three years
    but at least one full frequency-year, use an expanding percentile.

    Both branches are point-in-time safe and never use future rows.
    """

    source = pd.to_numeric(
        series,
        errors="coerce",
    ).sort_index()

    primary = rolling_percentile(
        source,
        frequency,
    )

    periods = periods_per_year(
        frequency
    )

    minimum = max(
        20,
        periods
        * fallback_years,
    )

    def last_percentile(
        values: np.ndarray,
    ) -> float:
        finite = values[
            np.isfinite(
                values
            )
        ]

        if len(finite) == 0:
            return np.nan

        latest_value = finite[-1]

        return (
            100.0
            * (
                np.sum(
                    finite
                    < latest_value
                )
                + 0.5
                * np.sum(
                    finite
                    == latest_value
                )
            )
            / len(
                finite
            )
        )

    fallback = (
        source
        .expanding(
            min_periods=minimum
        )
        .apply(
            last_percentile,
            raw=True,
        )
    )

    return primary.combine_first(
        fallback
    )


def inverse_percentile_score(pct: pd.Series) -> pd.Series:
    return (100.0 - pct).clip(0, 100)


def logistic_score(series: pd.Series, midpoint: float, scale: float, increasing: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    sign = 1.0 if increasing else -1.0
    z = np.clip(sign * (s - midpoint) / max(scale, 1e-9), -20, 20)
    return pd.Series(100.0 / (1.0 + np.exp(-z)), index=s.index).clip(0, 100)


def weighted_frame(scores: pd.DataFrame, weights: Mapping[str, float],
                   quality: Optional[pd.DataFrame] = None) -> Tuple[pd.Series, pd.Series]:
    frame = scores.reindex(columns=list(weights))
    valid = frame.notna()
    w = pd.Series(weights, dtype=float)
    if quality is None:
        q = valid.astype(float)
    else:
        q = quality.reindex_like(frame).fillna(0).clip(0, 1).where(valid, 0)
    ew = q.mul(w, axis=1)
    denom = ew.sum(axis=1)
    score = frame.fillna(0).mul(ew).sum(axis=1).div(denom.replace(0, np.nan)).clip(0, 100)
    coverage = denom.div(w.sum()).clip(0, 1)
    return score, coverage


def safe_weighted_average(scores: Mapping[str, Optional[float]], weights: Mapping[str, float]) -> Tuple[Optional[float], Dict[str, float], float]:
    valid = {k: float(v) for k, v in scores.items() if v is not None and np.isfinite(v)}
    if not valid:
        return None, {}, 0.0
    total_avail = sum(weights[k] for k in valid)
    eff = {k: weights[k] / total_avail for k in valid}
    return float(np.clip(sum(valid[k] * eff[k] for k in valid), 0, 100)), eff, total_avail / sum(weights.values())


def effective_weights(frame: pd.DataFrame, weights: Mapping[str, float]) -> Dict[str, float]:
    if frame.empty:
        return {}
    row = frame.iloc[-1]
    available = [k for k in weights if k in row.index and pd.notna(row[k])]
    total = sum(weights[k] for k in available)
    return {} if total <= 0 else {k: round(weights[k] / total, 6) for k in available}


# ============================================================
# 4. Market and FRED data
# ============================================================


def latest_component_gap_labels(
    frame: pd.DataFrame,
    expected_weights: Mapping[str, float],
    date: pd.Timestamp,
) -> List[str]:
    """
    Expose component-level score gaps that module coverage alone would
    otherwise hide from the dashboard.
    """

    labels: List[str] = []

    for component in expected_weights:
        if component not in frame.columns:
            labels.append(
                f"{component}_score"
            )
            continue

        value = frame[
            component
        ].get(
            date
        )

        if pd.isna(
            value
        ):
            labels.append(
                f"{component}_score"
            )

    return labels


def flatten_yfinance(raw: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if raw.empty:
        return {}
    fields = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
    out: Dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        l0, l1 = set(map(str, raw.columns.get_level_values(0))), set(map(str, raw.columns.get_level_values(1)))
        if any(f in l0 for f in fields):
            out = {f: raw[f].copy() for f in fields if f in l0}
        elif any(f in l1 for f in fields):
            out = {f: raw.xs(f, level=1, axis=1).copy() for f in fields if f in l1}
        else:
            raise ValueError("Unknown yfinance MultiIndex layout")
    else:
        out = {f: raw[[f]].copy() for f in fields if f in raw.columns}
    for frame in out.values():
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        frame.sort_index(inplace=True)
        frame.columns = [str(c) for c in frame.columns]
    return out

def load_cboe_vix3m(
    logger: logging.Logger,
) -> pd.Series:
    """
    從 Cboe 官方 CSV 取得 VIX3M 歷史收盤值。
    """

    url = (
        "https://cdn.cboe.com/api/global/"
        "us_indices/daily_prices/"
        "VIX3M_History.csv"
    )

    try:
        session = requests_session()

        response = session.get(
            url,
            timeout=(10, 60),
        )

        response.raise_for_status()

        frame = pd.read_csv(
            StringIO(response.text)
        )

        frame.columns = [
            str(column).strip().upper()
            for column in frame.columns
        ]

        required = {
            "DATE",
            "CLOSE",
        }

        missing = required - set(
            frame.columns
        )

        if missing:
            raise ValueError(
                "VIX3M CSV missing columns: "
                f"{sorted(missing)}"
            )

        frame["DATE"] = pd.to_datetime(
            frame["DATE"],
            errors="coerce",
        )

        frame["CLOSE"] = pd.to_numeric(
            frame["CLOSE"],
            errors="coerce",
        )

        series = (
            frame
            .dropna(
                subset=[
                    "DATE",
                    "CLOSE",
                ]
            )
            .drop_duplicates(
                "DATE",
                keep="last",
            )
            .set_index(
                "DATE"
            )["CLOSE"]
            .sort_index()
        )

        series.index = (
            series.index
            .tz_localize(None)
        )

        series.name = "^VIX3M"

        return series

    except Exception as exc:
        logger.warning(
            "Cboe VIX3M fallback failed: %s",
            exc,
        )

        return pd.Series(
            dtype=float,
            name="^VIX3M",
        )


def load_yfinance_data(cfg: Config, logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    cfg.cache_directory.mkdir(parents=True, exist_ok=True)
    try:
        raw = yf.download(list(cfg.market_tickers), start=cfg.start_date, auto_adjust=False,
                          progress=False, group_by="column", threads=True)
        data = flatten_yfinance(raw)
        if "Close" not in data:
            raise RuntimeError("No Close data returned")
        vix3m_official = load_cboe_vix3m(
            logger
        )

        if not vix3m_official.empty:
            close_frame = data["Close"].copy()

            existing_vix3m = (
                close_frame["^VIX3M"]
                if "^VIX3M" in close_frame.columns
                else pd.Series(
                    np.nan,
                    index=close_frame.index,
                )
            )

            official_aligned = (
                vix3m_official
                .reindex(
                    close_frame.index
                )
            )

            # 優先採用 Cboe 官方值；
            # 官方缺值時才保留 Yahoo 值。
            close_frame["^VIX3M"] = (
                official_aligned
                .combine_first(
                    existing_vix3m
                )
            )

            data["Close"] = close_frame
        for field, frame in data.items():
            frame.to_csv(cfg.cache_directory / f"market_{field.replace(' ', '_').lower()}.csv")
        return data
    except Exception as exc:
        logger.warning("yfinance failed: %s; using cache", exc)
        data = {}
        for field in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
            p = cfg.cache_directory / f"market_{field.replace(' ', '_').lower()}.csv"
            if p.exists():
                data[field] = pd.read_csv(p, index_col=0, parse_dates=True)
        if "Close" not in data:
            raise RuntimeError("No market data and no cache") from exc
        return data


def validate_market_data(data: Dict[str, pd.DataFrame], cfg: Config,
                         logger: logging.Logger) -> Tuple[pd.DataFrame, List[str]]:
    close = data.get("Close")
    if close is None or close.empty:
        close = data.get("Adj Close")
    if close is None or close.empty or "SPY" not in close.columns:
        raise RuntimeError("SPY Close data is required")
    close = close.sort_index()
    missing = [t for t in cfg.market_tickers if t not in close.columns or close[t].dropna().empty]
    for ticker in missing:
        logger.warning("Missing market ticker: %s", ticker)
    calendar = close.index[close["SPY"].notna()]
    return close.reindex(calendar), missing


def fred_download(
    series_id: str,
    cfg: Config,
    session: requests.Session,
) -> pd.Series:
    """
    下載單一 FRED 序列。

    優先順序：
    1. 有 FRED_API_KEY：使用官方 JSON API。
    2. 沒有 API Key：使用公開 FRED CSV。
    3. 兩者失敗後，由 load_live_fred() 接手讀取本機快取。
    """

    if cfg.fred_api_key:
        url = (
            "https://api.stlouisfed.org/"
            "fred/series/observations"
        )

        params = {
            "series_id": series_id,
            "api_key": cfg.fred_api_key,
            "file_type": "json",
            "observation_start": cfg.start_date,
        }

        response = session.get(
            url,
            params=params,
            timeout=(10, 60),
        )

        response.raise_for_status()

        items = response.json().get("observations", [])

        records = {}

        for item in items:
            date_value = item.get("date")
            raw_value = item.get("value")

            if date_value is None:
                continue

            numeric_value = pd.to_numeric(
                raw_value,
                errors="coerce",
            )

            if pd.isna(numeric_value):
                continue

            records[pd.Timestamp(date_value)] = numeric_value

        result = pd.Series(
            records,
            name=series_id,
            dtype=float,
        ).sort_index()

        if result.empty:
            raise RuntimeError(
                f"FRED API returned empty data for {series_id}"
            )

        return result

    # 沒有 API Key 時的公開 CSV fallback
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    response = session.get(
        url,
        params={
            "id": series_id,
            "cosd": cfg.start_date,
        },
        timeout=(10, 90),
    )

    response.raise_for_status()

    frame = pd.read_csv(StringIO(response.text))

    if frame.empty or len(frame.columns) < 2:
        raise RuntimeError(
            f"FRED CSV returned invalid data for {series_id}"
        )

    date_column = frame.columns[0]

    value_column = (
        series_id
        if series_id in frame.columns
        else frame.columns[-1]
    )

    frame[date_column] = pd.to_datetime(
        frame[date_column],
        errors="coerce",
    )

    frame[value_column] = pd.to_numeric(
        frame[value_column],
        errors="coerce",
    )

    result = (
        frame
        .dropna(subset=[date_column, value_column])
        .set_index(date_column)[value_column]
        .sort_index()
    )

    if result.empty:
        raise RuntimeError(
            f"FRED CSV returned empty data for {series_id}"
        )

    result.name = series_id
    return result


def load_live_fred(cfg: Config, logger: logging.Logger) -> Dict[str, pd.Series]:
    cfg.cache_directory.mkdir(parents=True, exist_ok=True)
    session = requests_session()
    out = {}
    for sid in cfg.fred_metadata:
        cache = cfg.cache_directory / f"fred_{sid}.csv"
        try:
            s = fred_download(sid, cfg, session)
            if s.empty:
                raise RuntimeError("empty series")
            s.to_frame("value").to_csv(cache)
            out[sid] = s
        except Exception as exc:
            logger.warning("FRED %s failed: %s", sid, exc)
            if cache.exists():
                out[sid] = pd.to_numeric(pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0], errors="coerce").dropna()
            else:
                out[sid] = pd.Series(dtype=float, name=sid)
    return out


def load_strict_pit_macro(cfg: Config) -> Dict[str, pd.Series]:
    path = cfg.data_directory / "macro_vintages.csv"
    required = {"series_id", "observation_date", "release_timestamp", "effective_trade_date", "value", "source"}
    if not path.exists():
        raise FileNotFoundError("strict_pit requires data/macro_vintages.csv")
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"macro_vintages.csv missing {sorted(missing)}")
    frame["effective_trade_date"] = pd.to_datetime(frame["effective_trade_date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    out = {}
    for sid in cfg.fred_metadata:
        x = frame[frame["series_id"] == sid].dropna(subset=["effective_trade_date", "value"]).sort_values(["effective_trade_date", "release_timestamp"])
        out[sid] = x.groupby("effective_trade_date")["value"].last().sort_index() if not x.empty else pd.Series(dtype=float, name=sid)
    missing_sids = [sid for sid, s in out.items() if s.empty]
    if missing_sids:
        raise RuntimeError("strict_pit missing: " + ", ".join(missing_sids))
    return out


def to_event_series(observations: Dict[str, pd.Series], cfg: Config) -> Dict[str, pd.Series]:
    out = {}
    for sid, s in observations.items():
        meta = cfg.fred_metadata[sid]
        s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
        if s.empty:
            out[meta["alias"]] = pd.Series(dtype=float, name=meta["alias"])
            continue
        idx = s.index if cfg.data_mode == "strict_pit" else s.index + pd.to_timedelta(meta["lag_days"], unit="D")
        values = s.to_numpy(float)
        if meta["unit"] is not None:
            values *= float(meta["unit"])
        out[meta["alias"]] = pd.Series(values, index=idx, name=meta["alias"]).groupby(level=0).last().sort_index()
    return out


def align_event_series(events: pd.Series, calendar: pd.DatetimeIndex, max_age: int,
                       source: str, is_proxy: bool = False) -> AlignedSeries:
    calendar = pd.DatetimeIndex(calendar).sort_values()
    if events.empty:
        blank = pd.Series(np.nan, index=calendar)
        return AlignedSeries(blank, pd.Series(pd.NaT, index=calendar), blank.copy(), pd.Series(True, index=calendar), source, is_proxy)
    e = pd.DataFrame({"update_date": pd.to_datetime(events.index), "value": pd.to_numeric(events.values, errors="coerce")}).dropna().sort_values("update_date")
    target = pd.DataFrame({"date": calendar})
    a = pd.merge_asof(target, e, left_on="date", right_on="update_date", direction="backward").set_index("date")
    age = (a.index.to_series() - a["update_date"]).dt.days
    stale = a["update_date"].isna() | (age > max_age)
    return AlignedSeries(a["value"].where(~stale), a["update_date"], age, stale, source, is_proxy)

def align_live_public_event(
    events: pd.Series,
    calendar: pd.DatetimeIndex,
    max_age: int,
    source: str,
    cfg: Config,
    is_proxy: bool = False,
) -> pd.Series:
    """
    將事件資料依 effective_trade_date 對齊。

    live_public：
    最新報告使用 signal_date 收盤資料，
    但實際適用於 decision_date，因此允許把
    decision_date 已生效的最新資料映射到
    最新 signal_date 報告。

    不回填過去歷史日期。
    """

    signal_date = pd.Timestamp(
        calendar.max()
    ).tz_localize(None)

    decision_date = next_nyse_session_after(
        signal_date
    )

    extended_calendar = (
        calendar.union(
            pd.DatetimeIndex(
                [decision_date]
            )
        )
        .sort_values()
    )

    aligned = align_event_series(
        events,
        extended_calendar,
        max_age,
        source,
        is_proxy,
    ).value

    if (
        cfg.data_mode == "live_public"
        and decision_date > signal_date
    ):
        decision_value = aligned.get(
            decision_date
        )

        if pd.notna(decision_value):
            aligned.loc[
                signal_date
            ] = decision_value

    return aligned.reindex(
        calendar
    )


def apply_staleness_rules(events: Dict[str, pd.Series], calendar: pd.DatetimeIndex, cfg: Config) -> Dict[str, AlignedSeries]:
    by_alias = {m["alias"]: m for m in cfg.fred_metadata.values()}
    return {alias: align_event_series(s, calendar, cfg.max_staleness_days[by_alias[alias]["age"]], f"FRED:{alias}") for alias, s in events.items()}


def load_optional_csv(path: Path, required: Iterable[str], logger: logging.Logger) -> pd.DataFrame:
    if not path.exists():
        logger.warning("Optional file missing: %s", path)
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        logger.warning("Optional file empty: %s", path)
        return pd.DataFrame()
    missing = set(required) - set(frame.columns)
    if missing:
        logger.warning("%s missing columns %s", path, sorted(missing))
        return pd.DataFrame()
    for c in ("observation_date", "release_timestamp", "effective_trade_date", "event_date"):
        if c in frame:
            frame[c] = pd.to_datetime(frame[c], errors="coerce")
    for c in ("value", "score"):
        if c in frame:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
    if "is_proxy" in frame:
        frame["is_proxy"] = frame["is_proxy"].map(as_bool)
    return frame


# ============================================================
# 5. Optional PIT modules
# ============================================================

def event_metric_score(frame: pd.DataFrame, metric: str, high_good: bool, frequency: str) -> pd.Series:
    x = frame[frame["metric"].astype(str).str.lower() == metric.lower()].copy()
    if x.empty:
        return pd.Series(dtype=float, name=metric)
    x = x.sort_values(["effective_trade_date", "release_timestamp"])
    if "score" in x and x["score"].notna().any():
        return x.groupby("effective_trade_date")["score"].last().dropna().clip(0, 100)
    raw = x.groupby("effective_trade_date")["value"].last().dropna()
    pct = rolling_percentile(winsorize_series(raw, frequency), frequency, min_years=2)
    return (pct if high_good else inverse_percentile_score(pct)).dropna()


def calculate_valuation(
    calendar: pd.DatetimeIndex,
    cfg: Config,
    logger: logging.Logger,
) -> Tuple[
    ModuleResult,
    Optional[pd.Series],
]:
    blank = pd.Series(
        np.nan,
        index=calendar,
    )

    if cfg.data_mode == "market_only":
        return (
            ModuleResult(
                "valuation",
                blank,
                pd.Series(0.0, index=calendar),
                pd.DataFrame(index=calendar),
                {},
                False,
                ["disabled_in_market_only"],
                [],
                {},
            ),
            None,
        )

    required = {
        "observation_date",
        "release_timestamp",
        "effective_trade_date",
        "asset",
        "metric",
        "value",
        "source",
        "is_proxy",
    }

    frame = load_optional_csv(
        cfg.data_directory
        / "valuation_pit.csv",
        required,
        logger,
    )

    if frame.empty:
        if cfg.data_mode == "strict_pit":
            raise RuntimeError(
                "strict_pit requires valuation_pit.csv"
            )

        return (
            ModuleResult(
                "valuation",
                blank,
                pd.Series(0.0, index=calendar),
                pd.DataFrame(index=calendar),
                {},
                False,
                ["valuation_pit.csv"],
                [],
                {},
            ),
            None,
        )

    directions = {
        "trailing_pe": False,
        "fcf_yield": True,
        "earnings_yield": True,
        "erp": True,
    }

    base_weights = {
        "trailing_pe": 0.30,
        "fcf_yield": 0.25,
        "earnings_yield": 0.25,
        "erp": 0.20,
    }

    signal_date = pd.Timestamp(
        calendar.max()
    )

    decision_date = (
        next_nyse_session_after(
            signal_date
        )
    )

    # Valuation 可能從 decision_date 才生效，
    # 所以暫時將 decision_date 加入對齊日曆。
    valuation_calendar = (
        calendar.union(
            pd.DatetimeIndex(
                [decision_date]
            )
        )
        .sort_values()
    )

    components: Dict[str, pd.Series] = {}
    raw: Dict[str, pd.Series] = {}
    proxies: List[bool] = []

    for metric, high_good in directions.items():
        metric_rows = frame[
            frame["metric"]
            .astype(str)
            .str.lower()
            .eq(metric)
        ].copy()

        if metric_rows.empty:
            continue

        score_event = event_metric_score(
            frame,
            metric,
            high_good,
            "quarterly",
        )

        raw_event = (
            metric_rows
            .sort_values(
                [
                    "effective_trade_date",
                    "release_timestamp",
                ]
            )
            .groupby(
                "effective_trade_date"
            )["value"]
            .last()
            .dropna()
        )

        proxy = bool(
            metric_rows["is_proxy"]
            .fillna(False)
            .any()
        )

        proxies.append(proxy)

        aligned_score = align_event_series(
            score_event,
            valuation_calendar,
            cfg.max_staleness_days[
                "quarterly_fundamental"
            ],
            f"valuation:{metric}",
            proxy,
        ).value

        aligned_raw = align_event_series(
            raw_event,
            valuation_calendar,
            cfg.max_staleness_days[
                "quarterly_fundamental"
            ],
            f"valuation_raw:{metric}",
            proxy,
        ).value

        # live_public 報告是：
        # 用 signal_date 的收盤資料，
        # 做 decision_date 的交易決策。
        #
        # 因此 decision_date 已生效的估值，
        # 可以放入最新 signal_date 報告，
        # 但只限最新一筆，不能回填整段歷史。
        if (
            cfg.data_mode == "live_public"
            and decision_date > signal_date
        ):
            decision_score = (
                aligned_score.get(
                    decision_date
                )
            )

            decision_raw = (
                aligned_raw.get(
                    decision_date
                )
            )

            if pd.notna(decision_score):
                aligned_score.loc[
                    signal_date
                ] = decision_score

            if pd.notna(decision_raw):
                aligned_raw.loc[
                    signal_date
                ] = decision_raw

        components[metric] = (
            aligned_score.reindex(
                calendar
            )
        )

        raw[metric] = (
            aligned_raw.reindex(
                calendar
            )
        )

    if not components:
        return (
            ModuleResult(
                "valuation",
                blank,
                pd.Series(0.0, index=calendar),
                pd.DataFrame(index=calendar),
                {},
                False,
                ["usable_valuation_metrics"],
                [],
                {},
            ),
            None,
        )

    component_frame = pd.DataFrame(
        components,
        index=calendar,
    )

    score, coverage = weighted_frame(
        component_frame,
        base_weights,
    )

    missing_inputs = [
        key
        for key in base_weights
        if key not in component_frame.columns
    ]

    available_metrics = set(
        frame["metric"]
        .dropna()
        .astype(str)
        .str.lower()
    )

    if not {
        "forward_pe",
        "forward_earnings_yield",
    }.intersection(
        available_metrics
    ):
        missing_inputs.append(
            "forward_estimates_unavailable"
        )

    earnings_yield = raw.get(
        "earnings_yield"
    )

    return (
        ModuleResult(
            "valuation",
            score,
            coverage,
            component_frame,
            base_weights,
            any(proxies),
            sorted(set(missing_inputs)),
            [],
            raw,
        ),
        earnings_yield,
    )

def calculate_ai_cycle(
    calendar: pd.DatetimeIndex,
    cfg: Config,
    logger: logging.Logger,
) -> ModuleResult:
    blank = pd.Series(np.nan, index=calendar)

    if cfg.data_mode == "market_only":
        return ModuleResult(
            "ai_cycle",
            blank,
            pd.Series(0.0, index=calendar),
            pd.DataFrame(index=calendar),
            {},
            False,
            ["disabled_in_market_only"],
            [],
            {},
        )

    required = {
        "observation_date",
        "release_timestamp",
        "effective_trade_date",
        "company",
        "metric",
        "value",
        "source",
        "is_proxy",
    }

    frame = load_optional_csv(
        cfg.data_directory / "ai_cycle_pit.csv",
        required,
        logger,
    )

    if frame.empty:
        if cfg.data_mode == "strict_pit":
            raise RuntimeError("strict_pit requires ai_cycle_pit.csv")

        return ModuleResult(
            "ai_cycle",
            blank,
            pd.Series(0.0, index=calendar),
            pd.DataFrame(index=calendar),
            {},
            False,
            ["ai_cycle_pit.csv"],
            [],
            {},
        )

    weights0 = {
        "nvidia_revenue_yoy_proxy": 0.35,
        "hyperscaler_capex_yoy": 0.45,
        "tsmc_hpc_growth": 0.10,
        "micron_dc_hbm_score": 0.10,
    }

    components: Dict[str, pd.Series] = {}
    raw: Dict[str, pd.Series] = {}
    proxies: List[bool] = []

    for metric in weights0:
        metric_rows = frame[
            frame["metric"].astype(str).str.lower().eq(metric)
        ].copy()

        if metric_rows.empty:
            continue

        score_event = event_metric_score(
            frame,
            metric,
            True,
            "quarterly",
        )

        raw_event = (
            metric_rows
            .sort_values(["effective_trade_date", "release_timestamp"])
            .groupby("effective_trade_date")["value"]
            .last()
            .dropna()
        )

        proxy = bool(
            metric_rows["is_proxy"].fillna(False).any()
        )
        proxies.append(proxy)

        components[metric] = align_live_public_event(
            score_event,
            calendar,
            cfg.max_staleness_days["quarterly_fundamental"],
            f"ai:{metric}",
            cfg,
            proxy,
        )

        raw[metric] = align_live_public_event(
            raw_event,
            calendar,
            cfg.max_staleness_days["quarterly_fundamental"],
            f"ai_raw:{metric}",
            cfg,
            proxy,
        )

    if not components:
        return ModuleResult(
            "ai_cycle",
            blank,
            pd.Series(0.0, index=calendar),
            pd.DataFrame(index=calendar),
            weights0,
            False,
            ["usable_ai_metrics"],
            [],
            {},
        )

    component_frame = pd.DataFrame(
        components,
        index=calendar,
    )

    # Use the full configured weights so missing TSMC/Micron inputs
    # reduce coverage instead of being silently renormalized to 100%.
    missing_inputs = [
        key
        for key in weights0
        if key not in component_frame.columns
    ]

    missing_inputs.extend(
        latest_component_gap_labels(
            component_frame,
            weights0,
            calendar[-1],
        )
    )

    score, coverage = weighted_frame(
        component_frame,
        weights0,
    )

    return ModuleResult(
        "ai_cycle",
        score,
        coverage,
        component_frame,
        weights0,
        any(proxies),
        sorted(set(missing_inputs)),
        [],
        raw,
    )

def load_positioning_pit(
    calendar: pd.DatetimeIndex,
    cfg: Config,
    logger: logging.Logger,
) -> Tuple[
    Dict[str, pd.Series],
    Dict[str, pd.Series],
    Dict[str, bool],
]:
    required = {
        "observation_date",
        "release_timestamp",
        "effective_trade_date",
        "metric",
        "value",
        "source",
        "is_proxy",
    }

    frame = load_optional_csv(
        cfg.data_directory / "positioning_pit.csv",
        required,
        logger,
    )

    if frame.empty:
        return {}, {}, {}

    metric_config = {
        "equity_put_call": {
            "high_good": True,
            "frequency": "daily",
            "max_age": cfg.max_staleness_days["daily_market_indicator"],
        },
        "cftc_positioning": {
            "high_good": False,
            "frequency": "weekly",
            "max_age": cfg.max_staleness_days["weekly_macro"],
        },
        "etf_primary_flow": {
            "high_good": False,
            "frequency": "daily",
            "max_age": cfg.max_staleness_days["daily_market_indicator"],
        },
        "iv30": {
            "high_good": True,
            "frequency": "daily",
            "max_age": cfg.max_staleness_days["daily_market_indicator"],
        },
        "rv20": {
            "high_good": True,
            "frequency": "daily",
            "max_age": cfg.max_staleness_days["daily_market_indicator"],
        },
    }

    scores: Dict[str, pd.Series] = {}
    raw: Dict[str, pd.Series] = {}
    proxies: Dict[str, bool] = {}

    for metric, settings in metric_config.items():
        metric_rows = frame[
            frame["metric"].astype(str).str.lower().eq(metric)
        ].copy()

        if metric_rows.empty:
            continue

        metric_rows = metric_rows.sort_values(
            ["effective_trade_date", "release_timestamp"]
        )

        raw_event = (
            metric_rows
            .groupby("effective_trade_date")["value"]
            .last()
            .dropna()
        )

        proxy = bool(
            metric_rows["is_proxy"].fillna(False).any()
        )

        raw[metric] = align_live_public_event(
            raw_event,
            calendar,
            settings["max_age"],
            f"positioning_raw:{metric}",
            cfg,
            proxy,
        )

        if metric not in {"iv30", "rv20"}:
            score_event = event_metric_score(
                frame,
                metric,
                settings["high_good"],
                settings["frequency"],
            )

            scores[metric] = align_live_public_event(
                score_event,
                calendar,
                settings["max_age"],
                f"positioning:{metric}",
                cfg,
                proxy,
            )

        proxies[metric] = proxy

    return scores, raw, proxies


def load_nowcast(
    calendar: pd.DatetimeIndex,
    cfg: Config,
    logger: logging.Logger,
) -> Optional[pd.Series]:
    required = {
        "observation_date",
        "release_timestamp",
        "effective_trade_date",
        "metric",
        "value",
        "source",
        "is_proxy",
    }

    frame = load_optional_csv(
        cfg.data_directory / "macro_nowcast_pit.csv",
        required,
        logger,
    )

    if frame.empty:
        return None

    series: List[pd.Series] = []

    for metric in (
        frame["metric"]
        .dropna()
        .astype(str)
        .str.lower()
        .unique()
    ):
        event = event_metric_score(
            frame,
            metric,
            True,
            "weekly",
        )

        if event.empty:
            continue

        metric_rows = frame[
            frame["metric"].astype(str).str.lower().eq(metric)
        ]
        proxy = bool(
            metric_rows["is_proxy"].fillna(False).any()
        )

        series.append(
            align_live_public_event(
                event,
                calendar,
                cfg.max_staleness_days["weekly_macro"],
                f"nowcast:{metric}",
                cfg,
                proxy,
            )
        )

    if not series:
        return None

    return pd.concat(series, axis=1).mean(
        axis=1,
        skipna=True,
    )


# ============================================================
# 6. Core modules
# ============================================================


def calculate_market_regime(
    close: pd.DataFrame,
    fred: Dict[str, AlignedSeries],
    cfg: Config,
) -> ModuleResult:
    components: Dict[str, pd.Series] = {}
    raw: Dict[str, pd.Series] = {}
    missing: List[str] = []
    stale: List[str] = []

    for ticker, name in (
        ("SPY", "spy_trend"),
        ("QQQ", "qqq_trend"),
        ("SOXX", "soxx_trend"),
    ):
        if (
            ticker not in close.columns
            or close[ticker].dropna().empty
        ):
            missing.append(
                ticker
            )
            continue

        moving_average = (
            close[ticker]
            .rolling(
                200,
                min_periods=200,
            )
            .mean()
        )

        distance = (
            close[ticker]
            / moving_average
            - 1.0
        )

        distance = ewma_smooth(
            winsorize_series(
                distance,
                "daily",
            ),
            5,
        )

        components[
            name
        ] = rolling_percentile_with_fallback(
            distance,
            "daily",
        )

        raw[
            f"{name}_distance"
        ] = distance

    breadth_tickers = [
        ticker
        for ticker in (
            "SPY",
            "QQQ",
            "SOXX",
            "IWM",
            "MDY",
            "SMH",
        )
        if (
            ticker in close.columns
            and not close[
                ticker
            ].dropna().empty
        )
    ]

    if breadth_tickers:
        flags: Dict[str, pd.Series] = {}

        for ticker in breadth_tickers:
            moving_average = (
                close[ticker]
                .rolling(
                    200,
                    min_periods=200,
                )
                .mean()
            )

            flags[ticker] = (
                (
                    close[ticker]
                    > moving_average
                )
                .astype(float)
                .where(
                    moving_average.notna()
                )
            )

        breadth = ewma_smooth(
            pd.DataFrame(
                flags
            )
            .mean(
                axis=1,
                skipna=True,
            )
            * 100.0,
            5,
        )

        components[
            "breadth"
        ] = breadth

        raw[
            "breadth_score"
        ] = breadth
    else:
        missing.append(
            "breadth_proxy"
        )

    high_yield = fred.get(
        "hy_oas"
    )

    if (
        high_yield is None
        or high_yield.value.dropna().empty
    ):
        missing.append(
            "BAMLH0A0HYM2"
        )
    else:
        smooth = ewma_smooth(
            winsorize_series(
                high_yield.value,
                "daily",
            ),
            5,
        )

        percentile = (
            rolling_percentile_with_fallback(
                smooth,
                "daily",
            )
        )

        components[
            "credit"
        ] = inverse_percentile_score(
            percentile
        )

        raw[
            "hy_oas"
        ] = high_yield.value

        raw[
            "hy_oas_percentile"
        ] = percentile

        if (
            len(
                high_yield.stale
            ) > 0
            and bool(
                high_yield.stale.iloc[-1]
            )
        ):
            stale.append(
                "BAMLH0A0HYM2"
            )

    component_frame = pd.DataFrame(
        components,
        index=close.index,
    )

    missing.extend(
        latest_component_gap_labels(
            component_frame,
            cfg.market_regime_weights,
            close.index[-1],
        )
    )

    score, coverage = weighted_frame(
        component_frame,
        cfg.market_regime_weights,
    )

    return ModuleResult(
        "market_regime",
        score,
        coverage,
        component_frame,
        cfg.market_regime_weights,
        True,
        sorted(
            set(
                missing
            )
        ),
        sorted(
            set(
                stale
            )
        ),
        raw,
    )


def calculate_macro(
    calendar: pd.DatetimeIndex,
    events: Dict[str, pd.Series],
    aligned: Dict[str, AlignedSeries],
    nowcast: Optional[pd.Series],
    cfg: Config,
) -> ModuleResult:
    components: Dict[str, pd.Series] = {}
    raw: Dict[str, pd.Series] = {}
    missing: List[str] = []
    stale: List[str] = []

    core_pce = (
        events.get(
            "core_pce_index",
            pd.Series(
                dtype=float
            ),
        )
        .dropna()
    )

    if not core_pce.empty:
        annualized = (
            (
                core_pce
                / core_pce.shift(3)
            )
            ** 4
            - 1.0
        ) * 100.0

        deviation = (
            annualized
            - 2.0
        ).abs()

        score_event = (
            inverse_percentile_score(
                rolling_percentile_with_fallback(
                    deviation,
                    "monthly",
                )
            )
        )

        components[
            "core_pce"
        ] = align_event_series(
            score_event.dropna(),
            calendar,
            cfg.max_staleness_days[
                "monthly_macro"
            ],
            "core_pce",
        ).value

        raw[
            "core_pce_3m_annualized"
        ] = align_event_series(
            annualized.dropna(),
            calendar,
            cfg.max_staleness_days[
                "monthly_macro"
            ],
            "core_pce_raw",
        ).value
    else:
        missing.append(
            "PCEPILFE"
        )

    unemployment = (
        events.get(
            "unrate",
            pd.Series(
                dtype=float
            ),
        )
        .dropna()
    )

    if not unemployment.empty:
        average_3m = (
            unemployment
            .rolling(
                3,
                min_periods=3,
            )
            .mean()
        )

        gap = (
            average_3m
            - average_3m
            .rolling(
                12,
                min_periods=12,
            )
            .min()
        )

        score_event = (
            inverse_percentile_score(
                rolling_percentile_with_fallback(
                    gap,
                    "monthly",
                )
            )
        )

        components[
            "employment_gap"
        ] = align_event_series(
            score_event.dropna(),
            calendar,
            cfg.max_staleness_days[
                "monthly_macro"
            ],
            "sahm_gap",
        ).value

        raw[
            "sahm_style_gap"
        ] = align_event_series(
            gap.dropna(),
            calendar,
            cfg.max_staleness_days[
                "monthly_macro"
            ],
            "sahm_gap_raw",
        ).value
    else:
        missing.append(
            "UNRATE"
        )

    claims = (
        events.get(
            "claims",
            pd.Series(
                dtype=float
            ),
        )
        .dropna()
    )

    if not claims.empty:
        ratio = (
            claims
            .rolling(
                4,
                min_periods=4,
            )
            .mean()
            / claims
            .rolling(
                26,
                min_periods=26,
            )
            .mean()
            .replace(
                0,
                np.nan,
            )
        )

        score_event = (
            inverse_percentile_score(
                rolling_percentile_with_fallback(
                    ratio,
                    "weekly",
                )
            )
        )

        components[
            "initial_claims"
        ] = align_event_series(
            score_event.dropna(),
            calendar,
            cfg.max_staleness_days[
                "weekly_macro"
            ],
            "claims_ratio",
        ).value

        raw[
            "claims_4w_26w_ratio"
        ] = align_event_series(
            ratio.dropna(),
            calendar,
            cfg.max_staleness_days[
                "weekly_macro"
            ],
            "claims_ratio_raw",
        ).value
    else:
        missing.append(
            "ICSA"
        )

    curve_3m = (
        events.get(
            "t10y3m",
            pd.Series(
                dtype=float
            ),
        )
        .dropna()
    )

    curve_2y = (
        events.get(
            "t10y2y",
            pd.Series(
                dtype=float
            ),
        )
        .dropna()
    )

    if not curve_3m.empty:
        inversion = (
            -curve_3m
        ).clip(
            lower=0
        )

        prior_inversion = (
            curve_3m
            .rolling(
                126,
                min_periods=40,
            )
            .min()
            < 0
        )

        steepening = (
            curve_3m
            .diff(65)
            .clip(
                lower=0
            )
            .where(
                prior_inversion,
                0,
            )
        )

        stress = (
            inversion
            + 0.75
            * steepening
        )

        if not curve_2y.empty:
            stress = (
                0.75
                * stress
                + 0.25
                * (
                    -curve_2y
                    .reindex(
                        curve_3m.index
                    )
                    .ffill(
                        limit=7
                    )
                )
                .clip(
                    lower=0
                )
            )
        else:
            missing.append(
                "T10Y2Y"
            )

        score_event = (
            inverse_percentile_score(
                rolling_percentile_with_fallback(
                    stress,
                    "daily",
                )
            )
        )

        components[
            "yield_curve"
        ] = align_event_series(
            score_event.dropna(),
            calendar,
            cfg.max_staleness_days[
                "daily_market_indicator"
            ],
            "curve_stress",
            True,
        ).value

        raw[
            "yield_curve_stress"
        ] = align_event_series(
            stress.dropna(),
            calendar,
            cfg.max_staleness_days[
                "daily_market_indicator"
            ],
            "curve_stress_raw",
            True,
        ).value
    else:
        missing.append(
            "T10Y3M"
        )

    if (
        nowcast is not None
        and not nowcast.dropna().empty
    ):
        components[
            "nowcast_or_pmi"
        ] = nowcast

        raw[
            "gdpnow_score"
        ] = nowcast
    else:
        missing.append(
            "macro_nowcast_pit.csv"
        )

    for alias, series_id in (
        (
            "core_pce_index",
            "PCEPILFE",
        ),
        (
            "unrate",
            "UNRATE",
        ),
        (
            "claims",
            "ICSA",
        ),
        (
            "t10y3m",
            "T10Y3M",
        ),
    ):
        item = aligned.get(
            alias
        )

        if (
            item is not None
            and len(
                item.stale
            ) > 0
            and bool(
                item.stale.iloc[-1]
            )
        ):
            stale.append(
                series_id
            )

    component_frame = pd.DataFrame(
        components,
        index=calendar,
    )

    missing.extend(
        latest_component_gap_labels(
            component_frame,
            cfg.macro_weights,
            calendar[-1],
        )
    )

    score, coverage = weighted_frame(
        component_frame,
        cfg.macro_weights,
    )

    return ModuleResult(
        "macro",
        score,
        coverage,
        component_frame,
        cfg.macro_weights,
        True,
        sorted(
            set(
                missing
            )
        ),
        sorted(
            set(
                stale
            )
        ),
        raw,
    )

def calculate_liquidity(
    calendar: pd.DatetimeIndex,
    events: Dict[str, pd.Series],
    aligned: Dict[str, AlignedSeries],
    cfg: Config,
) -> ModuleResult:
    components: Dict[str, pd.Series] = {}
    raw: Dict[str, pd.Series] = {}
    missing: List[str] = []
    stale: List[str] = []

    def aligned_value(
        alias: str,
    ) -> pd.Series:
        item = aligned.get(alias)

        if item is None:
            return pd.Series(
                np.nan,
                index=calendar,
                dtype=float,
            )

        return (
            pd.to_numeric(
                item.value,
                errors="coerce",
            )
            .reindex(calendar)
        )

    # 三個 NFL 組成項全部對齊至同一個市場交易日日曆
    walcl_daily = aligned_value(
        "walcl_bn"
    )

    tga_daily = aligned_value(
        "tga_bn"
    )

    rrp_daily = aligned_value(
        "rrp_bn"
    )

    raw["walcl_bn"] = walcl_daily
    raw["tga_bn"] = tga_daily
    raw["rrp_bn"] = rrp_daily

    nfl_inputs_available = (
        walcl_daily.notna().any()
        and tga_daily.notna().any()
        and rrp_daily.notna().any()
    )

    if nfl_inputs_available:
        # 每一個日期都使用同一日期已知的
        # WALCL、TGA、RRP
        net_fed_liquidity = (
            walcl_daily
            - tga_daily
            - rrp_daily
        )

        # 13 週約 65 個交易日
        nfl_change_13w = (
            net_fed_liquidity.diff(65)
        )

        # 26 週約 130 個交易日
        nfl_change_26w = (
            net_fed_liquidity.diff(130)
        )

        nfl_percentile = (
            rolling_percentile_with_fallback(
                winsorize_series(
                    nfl_change_13w,
                    frequency="daily",
                ),
                frequency="daily",
            )
        )

        components[
            "nfl_13w_change"
        ] = nfl_percentile

        raw[
            "net_fed_liquidity_bn"
        ] = net_fed_liquidity

        raw[
            "nfl_13w_change_bn"
        ] = nfl_change_13w

        raw[
            "nfl_26w_change_bn"
        ] = nfl_change_26w

        raw[
            "nfl_13w_percentile"
        ] = nfl_percentile

        # 方便驗證公式是否一致
        raw[
            "nfl_formula_check_bn"
        ] = (
            net_fed_liquidity
            - (
                walcl_daily
                - tga_daily
                - rrp_daily
            )
        )
    else:
        if walcl_daily.dropna().empty:
            missing.append("WALCL")

        if tga_daily.dropna().empty:
            missing.append("WTREGEN")

        if rrp_daily.dropna().empty:
            missing.append("RRPONTSYD")

    # --------------------------------------------------------
    # Reserve balances
    # --------------------------------------------------------
    reserves = (
        events.get(
            "reserves_bn",
            pd.Series(dtype=float),
        )
        .dropna()
    )

    if not reserves.empty:
        reserve_change_13w = (
            reserves.diff(13)
        )

        reserve_score = (
            rolling_percentile_with_fallback(
                winsorize_series(
                    reserve_change_13w,
                    frequency="weekly",
                ),
                frequency="weekly",
            )
        )

        components[
            "reserve_balances"
        ] = align_event_series(
            reserve_score.dropna(),
            calendar,
            cfg.max_staleness_days[
                "weekly_macro"
            ],
            "reserves",
        ).value

        raw[
            "reserve_balances_bn"
        ] = aligned_value(
            "reserves_bn"
        )
    else:
        missing.append("WRESBAL")

    # --------------------------------------------------------
    # RRP inverse score
    # 使用真正每日 RRP 變化
    # --------------------------------------------------------
    rrp_events = (
        events.get(
            "rrp_bn",
            pd.Series(dtype=float),
        )
        .dropna()
    )

    if not rrp_events.empty:
        rrp_change_13w = (
            rrp_events.diff(65)
        )

        rrp_score = (
            inverse_percentile_score(
                rolling_percentile_with_fallback(
                    winsorize_series(
                        rrp_change_13w,
                        frequency="daily",
                    ),
                    frequency="daily",
                )
            )
        )

        components[
            "rrp_inverse"
        ] = align_event_series(
            rrp_score.dropna(),
            calendar,
            cfg.max_staleness_days[
                "daily_market_indicator"
            ],
            "rrp_inverse",
            True,
        ).value
    elif "RRPONTSYD" not in missing:
        missing.append("RRPONTSYD")

    # --------------------------------------------------------
    # TGA inverse score
    # TGA 仍屬週資料
    # --------------------------------------------------------
    tga_events = (
        events.get(
            "tga_bn",
            pd.Series(dtype=float),
        )
        .dropna()
    )

    if not tga_events.empty:
        tga_change_13w = (
            tga_events.diff(13)
        )

        tga_score = (
            inverse_percentile_score(
                rolling_percentile_with_fallback(
                    winsorize_series(
                        tga_change_13w,
                        frequency="weekly",
                    ),
                    frequency="weekly",
                )
            )
        )

        components[
            "tga_inverse"
        ] = align_event_series(
            tga_score.dropna(),
            calendar,
            cfg.max_staleness_days[
                "weekly_macro"
            ],
            "tga_inverse",
            True,
        ).value
    elif "WTREGEN" not in missing:
        missing.append("WTREGEN")

    # --------------------------------------------------------
    # Real M2
    # --------------------------------------------------------
    real_m2 = (
        events.get(
            "real_m2_bn",
            pd.Series(dtype=float),
        )
        .dropna()
    )

    if not real_m2.empty:
        real_m2_yoy = (
            real_m2.pct_change(12)
            * 100
        )

        real_m2_score = (
            rolling_percentile_with_fallback(
                winsorize_series(
                    real_m2_yoy,
                    frequency="monthly",
                ),
                frequency="monthly",
            )
        )

        components[
            "real_m2"
        ] = align_event_series(
            real_m2_score.dropna(),
            calendar,
            cfg.max_staleness_days[
                "monthly_macro"
            ],
            "real_m2",
        ).value

        raw[
            "real_m2_yoy"
        ] = align_event_series(
            real_m2_yoy.dropna(),
            calendar,
            cfg.max_staleness_days[
                "monthly_macro"
            ],
            "real_m2_raw",
        ).value
    else:
        missing.append("M2REAL")

    # --------------------------------------------------------
    # Staleness
    # --------------------------------------------------------
    for alias, series_id in (
        ("walcl_bn", "WALCL"),
        ("tga_bn", "WTREGEN"),
        ("rrp_bn", "RRPONTSYD"),
        ("reserves_bn", "WRESBAL"),
        ("real_m2_bn", "M2REAL"),
    ):
        item = aligned.get(alias)

        if (
            item is not None
            and len(item.stale) > 0
            and bool(item.stale.iloc[-1])
        ):
            stale.append(series_id)

    component_frame = pd.DataFrame(
        components,
        index=calendar,
    )

    missing.extend(
        latest_component_gap_labels(
            component_frame,
            cfg.liquidity_weights,
            calendar[-1],
        )
    )

    score, coverage = weighted_frame(
        component_frame,
        cfg.liquidity_weights,
    )

    return ModuleResult(
        "liquidity",
        score,
        coverage,
        component_frame,
        cfg.liquidity_weights,
        True,
        sorted(set(missing)),
        sorted(set(stale)),
        raw,
    )


def calculate_positioning(
    close: pd.DataFrame,
    ext_scores: Dict[str, pd.Series],
    ext_raw: Dict[str, pd.Series],
    proxies: Dict[str, bool],
    cfg: Config,
) -> ModuleResult:
    components: Dict[str, pd.Series] = {}
    raw: Dict[str, pd.Series] = {}
    missing: List[str] = []

    if (
        "^VIX" in close.columns
        and not close[
            "^VIX"
        ].dropna().empty
    ):
        vix = ewma_smooth(
            winsorize_series(
                close[
                    "^VIX"
                ],
                "daily",
            ),
            5,
        )

        percentile = (
            rolling_percentile_with_fallback(
                vix,
                "daily",
            )
        )

        components[
            "vix_percentile"
        ] = percentile

        raw[
            "vix"
        ] = close[
            "^VIX"
        ]

        raw[
            "vix_percentile"
        ] = percentile
    else:
        missing.append(
            "^VIX"
        )

    if (
        "^VIX" in close.columns
        and "^VIX3M" in close.columns
        and not close[
            "^VIX3M"
        ].dropna().empty
    ):
        ratio = (
            close[
                "^VIX"
            ]
            / close[
                "^VIX3M"
            ].replace(
                0,
                np.nan,
            )
        )

        components[
            "vix_term_structure"
        ] = logistic_score(
            ratio,
            1.0,
            0.035,
            increasing=False,
        )

        raw[
            "vix3m"
        ] = close[
            "^VIX3M"
        ]

        raw[
            "vix_term_ratio"
        ] = ratio

        raw[
            "vix_backwardation"
        ] = (
            ratio
            > 1.0
        ).astype(float)
    else:
        missing.append(
            "^VIX3M"
        )

    if (
        "^SKEW" in close.columns
        and not close[
            "^SKEW"
        ].dropna().empty
    ):
        skew = ewma_smooth(
            winsorize_series(
                close[
                    "^SKEW"
                ],
                "daily",
            ),
            5,
        )

        components[
            "skew"
        ] = rolling_percentile_with_fallback(
            skew,
            "daily",
        )

        raw[
            "skew"
        ] = close[
            "^SKEW"
        ]

        raw[
            "skew_percentile"
        ] = components[
            "skew"
        ]
    else:
        missing.append(
            "^SKEW"
        )

    if (
        "^VVIX" in close.columns
        and not close[
            "^VVIX"
        ].dropna().empty
    ):
        vvix = ewma_smooth(
            winsorize_series(
                close[
                    "^VVIX"
                ],
                "daily",
            ),
            5,
        )

        components[
            "vvix"
        ] = rolling_percentile_with_fallback(
            vvix,
            "daily",
        )

        raw[
            "vvix"
        ] = close[
            "^VVIX"
        ]

        raw[
            "vvix_percentile"
        ] = components[
            "vvix"
        ]
    else:
        missing.append(
            "^VVIX"
        )

    if "equity_put_call" in ext_scores:
        components[
            "put_call"
        ] = ext_scores[
            "equity_put_call"
        ]

        raw[
            "equity_put_call"
        ] = ext_raw[
            "equity_put_call"
        ]
    else:
        missing.append(
            "equity_put_call"
        )

    positioning_sources = [
        ext_scores[
            metric
        ]
        for metric in (
            "cftc_positioning",
            "etf_primary_flow",
        )
        if metric in ext_scores
    ]

    if positioning_sources:
        components[
            "cftc_or_flow"
        ] = (
            pd.concat(
                positioning_sources,
                axis=1,
            )
            .mean(
                axis=1,
                skipna=True,
            )
        )

        if (
            "cftc_positioning"
            in ext_raw
        ):
            raw[
                "cftc_positioning"
            ] = ext_raw[
                "cftc_positioning"
            ]
    else:
        missing.append(
            "cftc_or_flow"
        )

    for metric in (
        "iv30",
        "rv20",
    ):
        if metric in ext_raw:
            raw[
                metric
            ] = ext_raw[
                metric
            ]
        else:
            missing.append(
                metric
            )

    if (
        "iv30" in raw
        and "rv20" in raw
    ):
        raw[
            "iv_rv_spread"
        ] = (
            raw[
                "iv30"
            ]
            - raw[
                "rv20"
            ]
        )

    component_frame = pd.DataFrame(
        components,
        index=close.index,
    )

    missing.extend(
        latest_component_gap_labels(
            component_frame,
            cfg.positioning_weights,
            close.index[-1],
        )
    )

    score, coverage = weighted_frame(
        component_frame,
        cfg.positioning_weights,
    )

    return ModuleResult(
        "positioning",
        score,
        coverage,
        component_frame,
        cfg.positioning_weights,
        any(
            proxies.values()
        ),
        sorted(
            set(
                missing
            )
        ),
        [],
        raw,
    )

# ============================================================
# 7. Decision scores and actions
# ============================================================

def combine_modules(modules: Mapping[str, ModuleResult], weights: Mapping[str, float]) -> Tuple[pd.Series, pd.Series]:
    scores = pd.DataFrame({k: v.score for k, v in modules.items()})
    quality = pd.DataFrame({k: v.coverage for k, v in modules.items()})
    return weighted_frame(scores, weights, quality)


def calculate_buy_score(modules: Mapping[str, ModuleResult], cfg: Config) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    base, coverage = combine_modules(modules, cfg.buy_score_weights)
    oas = modules["market_regime"].raw.get("hy_oas_percentile", pd.Series(np.nan, index=base.index))
    ratio = modules["positioning"].raw.get("vix_term_ratio", pd.Series(np.nan, index=base.index))
    vix = modules["positioning"].raw.get("vix", pd.Series(np.nan, index=base.index))
    severe = (ratio > 1) & (vix > 30)
    bonus = ((modules["positioning"].score >= 80) & (oas < 90) & (modules["liquidity"].score >= 40) & ~severe.fillna(False)).astype(float) * 10
    valuation = modules["valuation"].score
    penalty = ((modules["positioning"].score <= 20) & valuation.notna() & (valuation <= 40)).astype(float) * 10
    return (base + bonus - penalty).clip(0, 100), coverage, bonus, penalty


def calculate_risk_score(modules: Mapping[str, ModuleResult], buy_coverage: pd.Series, cfg: Config) -> Tuple[pd.Series, pd.Series, Dict[str, pd.Series]]:
    support, coverage = combine_modules(modules, cfg.risk_support_weights)
    risk = 100 - support
    idx = risk.index
    raw_mr, raw_pos, raw_liq = modules["market_regime"].raw, modules["positioning"].raw, modules["liquidity"].raw
    overrides = {
        "HY_OAS_Extreme": (raw_mr.get("hy_oas_percentile", pd.Series(np.nan, index=idx)) > 90).fillna(False),
        "VIX_Backwardation_Spike": ((raw_pos.get("vix", pd.Series(np.nan, index=idx)) > 30) & (raw_pos.get("vix_term_ratio", pd.Series(np.nan, index=idx)) > 1)).fillna(False),
        "NFL_Drying_Up": (raw_liq.get("nfl_13w_percentile", pd.Series(np.nan, index=idx)) < 10).fillna(False),
        "Breadth_Collapse": (raw_mr.get("breadth_score", pd.Series(np.nan, index=idx)) < 20).fillna(False),
        "Low_Data_Coverage": (buy_coverage < .60).fillna(True),
        "Critical_Module_Unavailable": modules["market_regime"].score.isna() | modules["liquidity"].score.isna(),
    }
    for condition in overrides.values():
        risk += condition.astype(float) * 10
    return risk.clip(0, 100), coverage, overrides


def normalize_yield(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    med = s.abs().median(skipna=True)
    return s * 100 if pd.notna(med) and med <= 1 else s


def calculate_margin_score(
    modules: Mapping[str, ModuleResult],
    risk: pd.Series,
    earnings_yield: Optional[pd.Series],
    real_10y: pd.Series,
    overall_coverage: pd.Series,
    cfg: Config,
) -> Tuple[
    pd.Series,
    pd.Series,
    Optional[pd.Series],
    Optional[pd.Series],
]:
    base, coverage = combine_modules(
        modules,
        cfg.margin_base_weights,
    )

    carry = None
    spread = None

    if earnings_yield is not None:
        # earnings_yield 與 FRED DFII10
        # 都是百分點單位，例如 2.98、2.31，
        # 不可再透過 normalize_yield() 乘以 100。
        earnings = (
            pd.to_numeric(
                earnings_yield,
                errors="coerce",
            )
            .reindex(base.index)
        )

        real_yield = (
            pd.to_numeric(
                real_10y,
                errors="coerce",
            )
            .reindex(base.index)
        )

        borrow_rate_pct = (
            cfg.annual_borrow_rate
            * 100
        )

        spread = (
            earnings
            - real_yield
            - borrow_rate_pct
        )

        carry = logistic_score(
            spread,
            midpoint=0,
            scale=2,
            increasing=True,
        )

        blended_score = (
            0.85 * base
            + 0.15 * carry
        )

        score = blended_score.where(
            carry.notna(),
            base,
        )
    else:
        score = base.copy()

    score = score.where(
        risk <= 70,
        np.minimum(score, 40),
    )

    score = score.where(
        modules["liquidity"].score >= 30,
        np.minimum(score, 35),
    )

    score = score.where(
        overall_coverage >= 0.60,
        np.minimum(score, 40),
    )

    return (
        score.clip(0, 100),
        coverage,
        carry,
        spread,
    )   


def linear_map(x: float, a: float, b: float, c: float, d: float) -> float:
    return float(c + np.clip((x - a) / (b - a), 0, 1) * (d - c))


def determine_target_margin(buy: float, risk: float, margin: float, liquidity: float,
                            coverage: float, market_ok: bool, liquidity_ok: bool,
                            critical_stale: bool, valuation_ok: bool, carry_ok: bool,
                            cfg: Config) -> Tuple[float, List[str]]:
    stops = {
        "Buy_Score_below_45": buy < 45, "Risk_Score_above_70": risk > 70,
        "Margin_Score_below_45": margin < 45, "Liquidity_below_30": liquidity < 30,
        "Coverage_below_60pct": coverage < cfg.minimum_coverage_for_margin,
        "Market_Regime_unavailable": not market_ok, "Liquidity_unavailable": not liquidity_ok,
        "Critical_data_stale": critical_stale,
    }
    triggered = [k for k, v in stops.items() if v]
    if triggered:
        return 0.0, triggered
    if margin >= 80 and buy >= 85 and risk < 30 and liquidity >= 65 and coverage >= .80:
        target, reasons = linear_map(margin, 80, 100, 15, 20), ["Rare_high_conviction_margin_zone"]
    elif margin >= 70 and buy >= 75 and risk < 45 and liquidity >= 50:
        target, reasons = linear_map(margin, 70, 80, 10, 15), ["Aggressive_accumulation_zone"]
    elif margin >= 60 and buy >= 60 and risk < 60 and liquidity >= 40:
        target, reasons = linear_map(margin, 60, 70, 5, 10), ["Normal_accumulation_zone"]
    elif margin >= 45 and buy >= 45 and risk <= 70:
        target, reasons = linear_map(margin, 45, 60, 0, 5), ["Small_ETF_accumulation_zone"]
    else:
        target, reasons = 0.0, ["Neutral_no_margin_zone"]
    if not valuation_ok or not carry_ok:
        target = min(target, cfg.max_margin_without_valuation_or_carry)
        reasons.append("Margin_capped_without_valuation_or_carry")
    return round(float(np.clip(target, 0, cfg.max_target_margin_pct)), 1), reasons


def determine_target_cash(buy: float, risk: float) -> Tuple[int, int]:
    if buy < 45 or risk > 75: return 30, 40
    if 45 <= buy < 60 and 60 <= risk <= 75: return 20, 25
    if 60 <= buy < 75 and 45 <= risk < 60: return 10, 15
    if 75 <= buy < 85 and 30 <= risk < 45: return 8, 12
    if buy >= 85 and risk < 30: return 10, 10
    return 15, 25



def get_critical_stale_inputs(
    market_regime: ModuleResult,
    liquidity: ModuleResult,
) -> List[str]:
    """
    Return only stale inputs that should act as a hard leverage stop.

    Slower secondary indicators such as Real M2 may lower module
    coverage, but do not by themselves force target margin to zero.
    """

    critical_market_series = {
        "BAMLH0A0HYM2",
    }

    critical_liquidity_series = {
        "WALCL",
        "WTREGEN",
        "RRPONTSYD",
        "WRESBAL",
    }

    stale = (
        critical_market_series.intersection(
            market_regime.stale_inputs
        )
        | critical_liquidity_series.intersection(
            liquidity.stale_inputs
        )
    )

    return sorted(stale)



def load_events(
    cfg: Config,
    logger: logging.Logger,
) -> Tuple[
    pd.DataFrame,
    bool,
    Dict[str, Any],
]:
    required = {
        "event_date",
        "asset",
        "event_type",
        "description",
        "source",
    }

    frame = load_optional_csv(
        cfg.data_directory
        / "events.csv",
        required,
        logger,
    )

    if frame.empty:
        return (
            frame,
            False,
            {
                "status":
                    "unavailable",
                "missing_checks":
                    [
                        "events.csv",
                    ],
            },
        )

    assets = set(
        frame[
            "asset"
        ]
        .dropna()
        .astype(str)
        .str.upper()
    )

    sources = set(
        frame[
            "source"
        ]
        .dropna()
        .astype(str)
        .str.upper()
    )

    descriptions = (
        frame[
            "description"
        ]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    event_types = (
        frame[
            "event_type"
        ]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    required_scopes = {
        "MARKET",
        "NVDA",
        "MSFT",
        "META",
        "GOOGL",
        "AMZN",
    }

    checks = {
        "required_asset_scopes":
            required_scopes.issubset(
                assets
            ),
        "fomc_calendar":
            bool(
                event_types.isin(
                    [
                        "fomc",
                        "fed",
                    ]
                ).any()
            ),
        "bls_major_calendar":
            bool(
                (
                    frame[
                        "source"
                    ]
                    .astype(str)
                    .str.upper()
                    .eq(
                        "BLS_ICS"
                    )
                    & descriptions.str.contains(
                        r"consumer price index"
                        r"|employment situation"
                        r"|producer price index",
                        regex=True,
                        na=False,
                    )
                ).any()
            ),
        "bea_gdp_or_pce_calendar":
            bool(
                (
                    frame[
                        "source"
                    ]
                    .astype(str)
                    .str.upper()
                    .eq(
                        "BEA_RELEASE_SCHEDULE"
                    )
                    & descriptions.str.contains(
                        r"gross domestic product"
                        r"|personal income and outlays"
                        r"|gdp",
                        regex=True,
                        na=False,
                    )
                ).any()
            ),
        "five_company_earnings":
            {
                "NVDA",
                "MSFT",
                "META",
                "GOOGL",
                "AMZN",
            }.issubset(
                set(
                    frame.loc[
                        event_types.eq(
                            "earnings"
                        ),
                        "asset",
                    ]
                    .dropna()
                    .astype(str)
                    .str.upper()
                )
            ),
    }

    missing_checks = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    complete = not missing_checks

    if not complete:
        logger.warning(
            "Event calendar is only partially "
            "available; missing checks: %s",
            missing_checks,
        )

    quality = {
        "status":
            (
                "complete"
                if complete
                else "partial"
            ),
        "checks":
            checks,
        "missing_checks":
            missing_checks,
        "sources":
            sorted(
                sources
            ),
        "asset_scopes":
            sorted(
                assets
            ),
    }

    return (
        frame,
        complete,
        quality,
    )

def upcoming_events(
    events: pd.DataFrame,
    date: pd.Timestamp,
    calendar: pd.DatetimeIndex,
    n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Return only events important enough to affect leverage or
    covered-call decisions within the next n NYSE sessions.
    """

    if events.empty:
        return []

    frame = events.copy()

    event_type = (
        frame["event_type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    description = (
        frame["description"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    major_macro_pattern = (
        r"consumer price index"
        r"|employment situation"
        r"|producer price index"
        r"|employment cost index"
        r"|personal income and outlays"
        r"|personal consumption expenditures"
        r"|gross domestic product"
    )

    major_mask = (
        event_type.eq("earnings")
        | event_type.isin(
            {
                "fomc",
                "fed",
            }
        )
        | (
            event_type.eq(
                "macro_release"
            )
            & description.str.contains(
                major_macro_pattern,
                regex=True,
                na=False,
            )
        )
    )

    frame = frame.loc[
        major_mask
    ].copy()

    if frame.empty:
        return []

    signal_date = pd.Timestamp(
        date
    ).tz_localize(None)

    nyse = mcal.get_calendar(
        "NYSE"
    )

    schedule = nyse.schedule(
        start_date=(
            signal_date
            + pd.Timedelta(days=1)
        ).date(),
        end_date=(
            signal_date
            + pd.Timedelta(days=45)
        ).date(),
    )

    if schedule.empty:
        return []

    future_sessions = pd.DatetimeIndex(
        schedule.index
    )[:n]

    if future_sessions.empty:
        return []

    last_date = pd.Timestamp(
        future_sessions[-1]
    ).tz_localize(None)

    event_dates = pd.to_datetime(
        frame["event_date"],
        errors="coerce",
    )

    mask = (
        event_dates.gt(
            signal_date
        )
        & event_dates.le(
            last_date
        )
    )

    return (
        frame.loc[mask]
        .assign(
            event_date=event_dates.loc[
                mask
            ]
        )
        .drop_duplicates(
            subset=[
                "event_date",
                "asset",
                "event_type",
                "description",
            ],
            keep="last",
        )
        .sort_values(
            [
                "event_date",
                "asset",
            ]
        )
        .to_dict(
            "records"
        )
    )


def determine_cc(buy: float, risk: float, coverage: float, panic_bonus: float,
                 ai: Optional[float], valuation: Optional[float], events: List[Dict[str, Any]],
                 event_check: bool, critical_ok: bool, iv_available: bool,
                 iv_positive: Optional[bool]) -> Tuple[str, List[str]]:
    reasons = []
    if buy >= 75: reasons.append("Buy_Score_high_preserve_upside")
    if risk > 70: reasons.append("Risk_Score_high")
    if panic_bonus > 0: reasons.append("Panic_rebound_signal")
    if ai is not None and ai >= 75: reasons.append("AI_Cycle_accelerating")
    if coverage < .60: reasons.append("Low_data_coverage")
    if events: reasons.append("Major_event_within_10_trading_days")
    if not critical_ok: reasons.append("Critical_data_unavailable_or_stale")
    if reasons: return "STOP_NEW_CC", reasons
    if not event_check: return "HOLD_EXISTING_CC", ["Event_check_unavailable"]
    if 45 <= buy <= 59 and 60 <= risk <= 75: return "ALLOW_SMALL_CC", ["Moderate_buy_and_elevated_risk"]
    if 45 <= buy <= 65 and risk < 60 and valuation is not None and valuation < 45 and iv_available and iv_positive:
        return "NORMAL_CC", ["Valuation_expensive_and_IV_premium_positive"]
    return "HOLD_EXISTING_CC", ["Normal_CC_conditions_not_fully_confirmed"]


# ============================================================
# 8. Output and tests
# ============================================================

def latest(series: Optional[pd.Series], date: pd.Timestamp, digits: int = 4) -> Optional[float]:
    return None if series is None else number(series.get(date), digits)


def action_text(target: float, cash: Tuple[int, int]) -> str:
    if target <= 0: return f"禁止新增融資；建議現金 {cash[0]}%–{cash[1]}%，保留核心持股與風險彈性。"
    if target <= 5: return f"僅限核心 ETF 小量分批；目標融資 {target:.1f}%，現金 {cash[0]}%–{cash[1]}%。"
    if target <= 10: return f"常規分批加碼；目標融資 {target:.1f}%，現金 {cash[0]}%–{cash[1]}%。"
    if target <= 15: return f"積極分批加碼；目標融資 {target:.1f}%，現金 {cash[0]}%–{cash[1]}%。"
    return f"少見高信心區；目標融資 {target:.1f}%，仍保留至少 10% 現金。"



def build_output(
    cfg: Config,
    date: pd.Timestamp,
    modules: Mapping[str, ModuleResult],
    buy: pd.Series,
    risk: pd.Series,
    margin: pd.Series,
    buy_cov: pd.Series,
    risk_cov: pd.Series,
    margin_cov: pd.Series,
    overall_cov: pd.Series,
    bonus: pd.Series,
    penalty: pd.Series,
    carry: Optional[pd.Series],
    spread: Optional[pd.Series],
    overrides: Mapping[str, pd.Series],
    target: float,
    cash: Tuple[int, int],
    cc: str,
    cc_reasons: List[str],
    allocation_reasons: List[str],
    event_check: bool,
    event_quality: Dict[str, Any],
    events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    module_scores = {
        key: latest(
            result.score,
            date,
            2,
        )
        for key, result
        in modules.items()
    }

    stale = sorted(
        {
            item
            for result in modules.values()
            for item in result.stale_inputs
        }
    )

    missing = normalize_missing_labels(
        item
        for result in modules.values()
        for item in result.missing_inputs
    )

    proxy = sorted(
        key
        for key, result
        in modules.items()
        if result.is_proxy
    )

    unavailable = sorted(
        key
        for key, value
        in module_scores.items()
        if value is None
    )

    raw = {
        f"{module_name}.{indicator}":
            latest(
                series,
                date,
            )
        for module_name, result
        in modules.items()
        for indicator, series
        in result.raw.items()
    }

    critical_stale_series = (
        get_critical_stale_inputs(
            modules[
                "market_regime"
            ],
            modules[
                "liquidity"
            ],
        )
    )

    critical_ok = (
        module_scores[
            "market_regime"
        ] is not None
        and module_scores[
            "liquidity"
        ] is not None
        and not critical_stale_series
        and (
            latest(
                overall_cov,
                date,
            )
            or 0
        )
        >= 0.60
    )

    module_frame = pd.DataFrame(
        {
            key:
                result.score
            for key, result
            in modules.items()
        }
    )

    decision_date = (
        next_nyse_session_after(
            date
        )
    )

    iv30 = raw.get(
        "positioning.iv30"
    )

    rv20 = raw.get(
        "positioning.rv20"
    )

    iv_rv_spread = raw.get(
        "positioning.iv_rv_spread"
    )

    options_available = (
        iv30 is not None
        and rv20 is not None
    )

    output = {
        "signal_date":
            date.date().isoformat(),
        "decision_date":
            decision_date
            .date()
            .isoformat(),
        "generated_at":
            datetime.now(
                timezone.utc
            )
            .isoformat(),
        "data_mode":
            cfg.data_mode,
        "historical_data_is_revised":
            cfg.data_mode
            == "live_public",
        "scores": {
            "Buy_Score":
                latest(
                    buy,
                    date,
                    2,
                ),
            "Risk_Score":
                latest(
                    risk,
                    date,
                    2,
                ),
            "Margin_Score":
                latest(
                    margin,
                    date,
                    2,
                ),
            "Carry_Score":
                latest(
                    carry,
                    date,
                    2,
                ),
        },
        "modules": {
            "Market_Regime":
                module_scores[
                    "market_regime"
                ],
            "AI_Cycle":
                module_scores[
                    "ai_cycle"
                ],
            "Valuation":
                module_scores[
                    "valuation"
                ],
            "Macro":
                module_scores[
                    "macro"
                ],
            "Liquidity":
                module_scores[
                    "liquidity"
                ],
            "Positioning":
                module_scores[
                    "positioning"
                ],
        },
        "allocation": {
            "Target_Margin_Pct":
                target,
            "Target_Cash_Min_Pct":
                cash[0],
            "Target_Cash_Max_Pct":
                cash[1],
        },
        "covered_call": {
            "status":
                cc,
            "reason":
                cc_reasons,
            "options_data_available":
                options_available,
            "iv30_pct":
                iv30,
            "rv20_pct":
                rv20,
            "iv_minus_rv_pct_points":
                iv_rv_spread,
            # yfinance does not provide verified delta.
            "strike_recommendation_available":
                False,
        },
        "action":
            action_text(
                target,
                cash,
            ),
        "effective_module_weights": {
            "buy":
                effective_weights(
                    module_frame
                    .loc[
                        :date
                    ]
                    .tail(1),
                    cfg.buy_score_weights,
                ),
            "risk":
                effective_weights(
                    module_frame
                    .loc[
                        :date
                    ]
                    .tail(1),
                    cfg.risk_support_weights,
                ),
            "margin":
                effective_weights(
                    module_frame
                    .loc[
                        :date
                    ]
                    .tail(1),
                    cfg.margin_base_weights,
                ),
        },
        "module_data_quality": {
            key:
                result.latest_quality(
                    date
                )
            for key, result
            in modules.items()
        },
        "data_quality": {
            "coverage_ratio":
                latest(
                    overall_cov,
                    date,
                    4,
                ),
            "buy_coverage":
                latest(
                    buy_cov,
                    date,
                    4,
                ),
            "risk_coverage":
                latest(
                    risk_cov,
                    date,
                    4,
                ),
            "margin_coverage":
                latest(
                    margin_cov,
                    date,
                    4,
                ),
            "critical_data_ok":
                critical_ok,
            "critical_stale_series":
                critical_stale_series,
            "stale_series":
                stale,
            "missing_series":
                missing,
            "proxy_modules":
                proxy,
            "unavailable_modules":
                unavailable,
            "event_check_available":
                event_check,
            "event_calendar":
                event_quality,
        },
        "risk_overrides": [
            key
            for key, series
            in overrides.items()
            if bool(
                series.get(
                    date,
                    False,
                )
            )
        ],
        "decision_reasons":
            allocation_reasons,
        "modifiers": {
            "panic_bonus":
                latest(
                    bonus,
                    date,
                    2,
                ),
            "euphoria_penalty":
                latest(
                    penalty,
                    date,
                    2,
                ),
            "expected_return_spread_pct_points":
                latest(
                    spread,
                    date,
                    4,
                ),
        },
        "upcoming_events":
            events,
        "raw_indicators":
            raw,
        "methodology_notes": [
            (
                "live_public FRED history may contain revisions "
                "and is not strict PIT."
                if cfg.data_mode
                == "live_public"
                else
                "strict_pit uses supplied effective dates."
            ),
            (
                "Free valuation uses SEC-derived TTM fundamentals "
                "with annual fallback; it does not fabricate "
                "forward consensus estimates."
            ),
            (
                "AI Cycle uses calibrated revenue and amount-weighted "
                "CapEx proxies; TSMC HPC and Micron HBM remain "
                "unavailable without reliable PIT inputs."
            ),
            (
                "Options IV/RV data is a QQQ yfinance proxy. "
                "No verified delta is available, so no exact "
                "covered-call strike is produced."
            ),
        ],
    }

    return json_safe(
        output
    )

def build_history(calendar: pd.DatetimeIndex, modules: Mapping[str, ModuleResult],
                  buy: pd.Series, risk: pd.Series, margin: pd.Series,
                  coverage: pd.Series, bonus: pd.Series, cfg: Config) -> pd.DataFrame:
    h = pd.DataFrame(index=calendar)
    h.index.name = "date"
    for name, col in (("market_regime", "Market_Regime"), ("ai_cycle", "AI_Cycle"),
                      ("valuation", "Valuation"), ("macro", "Macro"),
                      ("liquidity", "Liquidity"), ("positioning", "Positioning")):
        h[col] = modules[name].score
    h["Buy_Score"], h["Risk_Score"], h["Margin_Score"], h["Coverage_Ratio"] = buy, risk, margin, coverage
    h["HY_OAS_Percentile"] = modules["market_regime"].raw.get("hy_oas_percentile")
    h["Breadth_Score"] = modules["market_regime"].raw.get("breadth_score")
    h["VIX"] = modules["positioning"].raw.get("vix")
    h["VIX3M"] = modules["positioning"].raw.get("vix3m")
    h["VIX_Backwardation"] = modules["positioning"].raw.get("vix_backwardation")
    h["Net_Fed_Liquidity"] = modules["liquidity"].raw.get("net_fed_liquidity_bn")
    h["NFL_13W_Change"] = modules["liquidity"].raw.get("nfl_13w_change_bn")
    targets, cash_min, cash_max, cc = [], [], [], []
    for d in calendar:
        b, r, m, l, c = (number(x.get(d)) for x in (buy, risk, margin, modules["liquidity"].score, coverage))
        if None in (b, r, m, l, c):
            t, ca = 0.0, (30, 40)
        else:
            t, _ = determine_target_margin(b, r, m, l, c, pd.notna(modules["market_regime"].score.get(d)), pd.notna(modules["liquidity"].score.get(d)), False, pd.notna(modules["valuation"].score.get(d)), False, cfg)
            ca = determine_target_cash(b, r)
        targets.append(t); cash_min.append(ca[0]); cash_max.append(ca[1]); cc.append("STOP_NEW_CC" if b is None or r is None or b >= 75 or r > 70 or (bonus.get(d, 0) or 0) > 0 else "HOLD_EXISTING_CC")
    h["Target_Margin_Pct"], h["Target_Cash_Min_Pct"], h["Target_Cash_Max_Pct"], h["Covered_Call_Status"] = targets, cash_min, cash_max, cc
    h["historical_data_is_revised"] = cfg.data_mode == "live_public"
    return h


def save_outputs(cfg: Config, output: Dict[str, Any], history: pd.DataFrame,
                 modules: Mapping[str, ModuleResult], date: pd.Timestamp) -> None:
    cfg.output_directory.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_directory / "latest_signals.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    history.to_csv(cfg.output_directory / "score_history.csv")
    rows = []
    for name, result in modules.items():
        row = {"signal_date": date.date().isoformat(), "module": name, **result.latest_quality(date)}
        for key in ("effective_weights", "missing_inputs", "stale_inputs"):
            row[key] = json.dumps(row[key], ensure_ascii=False)
        rows.append(row)
    pd.DataFrame(rows).to_csv(cfg.output_directory / "module_data_quality.csv", index=False)
    pd.DataFrame([{"signal_date": date.date().isoformat(), **output["raw_indicators"]}]).to_csv(cfg.output_directory / "raw_indicator_snapshot.csv", index=False)



def run_tests(
    output: Dict[str, Any],
    claims: pd.Series,
    modules: Mapping[str, ModuleResult],
) -> None:
    all_scores = {
        **output[
            "scores"
        ],
        **output[
            "modules"
        ],
    }

    for name, value in all_scores.items():
        if value is not None:
            assert (
                0
                <= value
                <= 100
            ), name

    target = output[
        "allocation"
    ][
        "Target_Margin_Pct"
    ]

    assert (
        0
        <= target
        <= 20
    )

    risk_score = output[
        "scores"
    ][
        "Risk_Score"
    ]

    liquidity_score = output[
        "modules"
    ][
        "Liquidity"
    ]

    margin_score = output[
        "scores"
    ][
        "Margin_Score"
    ]

    coverage = output[
        "data_quality"
    ][
        "coverage_ratio"
    ]

    if (
        risk_score is not None
        and risk_score > 70
    ):
        assert target == 0

    if (
        liquidity_score is not None
        and liquidity_score < 30
    ):
        assert target == 0

    if (
        margin_score is not None
        and margin_score < 45
    ):
        assert target == 0

    if (
        coverage is not None
        and coverage < 0.60
    ):
        assert target == 0

    if not claims.empty:
        expected = max(
            0,
            len(
                claims.dropna()
            )
            - 3,
        )

        actual = (
            claims
            .dropna()
            .rolling(4)
            .mean()
            .notna()
            .sum()
        )

        assert actual == expected

    options = output[
        "covered_call"
    ]

    if options[
        "options_data_available"
    ]:
        assert (
            options[
                "iv30_pct"
            ]
            is not None
        )

        assert (
            options[
                "rv20_pct"
            ]
            is not None
        )

        # No verified delta exists in this free proxy.
        assert (
            options[
                "strike_recommendation_available"
            ]
            is False
        )

    assert (
        "qqq_5y"
        not in modules[
            "valuation"
        ].raw
    )

    assert (
        "nvda_relative_strength"
        not in modules[
            "ai_cycle"
        ].raw
    )

    assert number(
        0.0
    ) == 0.0

# ============================================================
# 9. Main
# ============================================================


def main() -> None:
    cfg = CFG
    validate_config(
        cfg
    )

    cfg.data_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg.cache_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = configure_logging(
        cfg
    )

    logger.info(
        "Starting Investment OS production "
        "engine; mode=%s",
        cfg.data_mode,
    )

    market = load_yfinance_data(
        cfg,
        logger,
    )

    close, _ = validate_market_data(
        market,
        cfg,
        logger,
    )

    calendar = close.index
    date = close.index[-1]

    fred_observations = (
        load_strict_pit_macro(
            cfg
        )
        if cfg.data_mode
        == "strict_pit"
        else load_live_fred(
            cfg,
            logger,
        )
    )

    fred_events = to_event_series(
        fred_observations,
        cfg,
    )

    aligned = apply_staleness_rules(
        fred_events,
        calendar,
        cfg,
    )

    valuation, earnings_yield = (
        calculate_valuation(
            calendar,
            cfg,
            logger,
        )
    )

    ai_cycle = calculate_ai_cycle(
        calendar,
        cfg,
        logger,
    )

    (
        external_scores,
        external_raw,
        proxy_flags,
    ) = load_positioning_pit(
        calendar,
        cfg,
        logger,
    )

    nowcast = load_nowcast(
        calendar,
        cfg,
        logger,
    )

    market_regime = (
        calculate_market_regime(
            close,
            aligned,
            cfg,
        )
    )

    macro = calculate_macro(
        calendar,
        fred_events,
        aligned,
        nowcast,
        cfg,
    )

    liquidity = calculate_liquidity(
        calendar,
        fred_events,
        aligned,
        cfg,
    )

    positioning = calculate_positioning(
        close,
        external_scores,
        external_raw,
        proxy_flags,
        cfg,
    )

    modules = {
        "market_regime":
            market_regime,
        "ai_cycle":
            ai_cycle,
        "valuation":
            valuation,
        "macro":
            macro,
        "liquidity":
            liquidity,
        "positioning":
            positioning,
    }

    (
        buy,
        buy_coverage,
        bonus,
        penalty,
    ) = calculate_buy_score(
        modules,
        cfg,
    )

    (
        risk,
        risk_coverage,
        overrides,
    ) = calculate_risk_score(
        modules,
        buy_coverage,
        cfg,
    )

    _, overall_coverage = (
        combine_modules(
            modules,
            cfg.module_base_weights,
        )
    )

    real_10y = aligned.get(
        "real_10y",
        AlignedSeries(
            pd.Series(
                np.nan,
                index=calendar,
            ),
            pd.Series(
                pd.NaT,
                index=calendar,
            ),
            pd.Series(
                np.nan,
                index=calendar,
            ),
            pd.Series(
                True,
                index=calendar,
            ),
            "missing",
        ),
    ).value

    (
        margin,
        margin_coverage,
        carry,
        spread,
    ) = calculate_margin_score(
        modules,
        risk,
        earnings_yield,
        real_10y,
        overall_coverage,
        cfg,
    )

    values = [
        latest(
            series,
            date,
        )
        for series in (
            buy,
            risk,
            margin,
            liquidity.score,
            overall_coverage,
        )
    ]

    if any(
        value is None
        for value in values
    ):
        target = 0.0
        allocation_reasons = [
            "Latest_critical_scores_incomplete",
        ]
        cash = (
            30,
            40,
        )
    else:
        (
            buy_value,
            risk_value,
            margin_value,
            liquidity_value,
            coverage_value,
        ) = values

        critical_stale_series = (
            get_critical_stale_inputs(
                market_regime,
                liquidity,
            )
        )

        target, allocation_reasons = (
            determine_target_margin(
                buy_value,
                risk_value,
                margin_value,
                liquidity_value,
                coverage_value,
                latest(
                    market_regime.score,
                    date,
                )
                is not None,
                latest(
                    liquidity.score,
                    date,
                )
                is not None,
                bool(
                    critical_stale_series
                ),
                latest(
                    valuation.score,
                    date,
                )
                is not None,
                latest(
                    carry,
                    date,
                )
                is not None,
                cfg,
            )
        )

        cash = determine_target_cash(
            buy_value,
            risk_value,
        )

    (
        event_frame,
        event_check,
        event_quality,
    ) = load_events(
        cfg,
        logger,
    )

    future_events = upcoming_events(
        event_frame,
        date,
        calendar,
        10,
    )

    iv30 = external_raw.get(
        "iv30"
    )

    rv20 = external_raw.get(
        "rv20"
    )

    iv_available = (
        iv30 is not None
        and rv20 is not None
        and latest(
            iv30,
            date,
        )
        is not None
        and latest(
            rv20,
            date,
        )
        is not None
    )

    iv_positive = (
        None
        if not iv_available
        else (
            float(
                latest(
                    iv30,
                    date,
                )
            )
            > float(
                latest(
                    rv20,
                    date,
                )
            )
        )
    )

    critical_stale_series = (
        get_critical_stale_inputs(
            market_regime,
            liquidity,
        )
    )

    critical_ok = (
        latest(
            market_regime.score,
            date,
        )
        is not None
        and latest(
            liquidity.score,
            date,
        )
        is not None
        and not critical_stale_series
        and (
            latest(
                overall_coverage,
                date,
            )
            or 0
        )
        >= 0.60
    )

    covered_call, cc_reasons = (
        determine_cc(
            values[0]
            or 0,
            (
                values[1]
                if values[1]
                is not None
                else 100
            ),
            values[4]
            or 0,
            latest(
                bonus,
                date,
            )
            or 0,
            latest(
                ai_cycle.score,
                date,
            ),
            latest(
                valuation.score,
                date,
            ),
            future_events,
            event_check,
            critical_ok,
            iv_available,
            iv_positive,
        )
    )

    output = build_output(
        cfg,
        date,
        modules,
        buy,
        risk,
        margin,
        buy_coverage,
        risk_coverage,
        margin_coverage,
        overall_coverage,
        bonus,
        penalty,
        carry,
        spread,
        overrides,
        target,
        cash,
        covered_call,
        cc_reasons,
        allocation_reasons,
        event_check,
        event_quality,
        future_events,
    )

    history = build_history(
        calendar,
        modules,
        buy,
        risk,
        margin,
        overall_coverage,
        bonus,
        cfg,
    )

    save_outputs(
        cfg,
        output,
        history,
        modules,
        date,
    )

    run_tests(
        output,
        fred_events.get(
            "claims",
            pd.Series(
                dtype=float
            ),
        ),
        modules,
    )

    logger.info(
        "Completed: Buy=%s Risk=%s "
        "Margin=%s Target=%s%% CC=%s",
        output[
            "scores"
        ][
            "Buy_Score"
        ],
        output[
            "scores"
        ][
            "Risk_Score"
        ],
        output[
            "scores"
        ][
            "Margin_Score"
        ],
        target,
        covered_call,
    )

if __name__ == "__main__":
    main()
