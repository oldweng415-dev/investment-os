from __future__ import annotations

"""Collect a free live-public SEC valuation proxy for Investment OS.

Output:
    data/valuation_pit.csv

The collector uses:
- SEC EDGAR Company Facts: latest standardized annual fundamentals
- Yahoo Finance: latest public market capitalization and price date
- FRED DFII10: latest 10-year real yield

It does not create forward estimates. All rows are marked is_proxy=true.
"""

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "valuation_pit.csv"
FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "").strip()

COMPANIES: dict[str, str] = {
    "NVDA": "0001045810",
    "MSFT": "0000789019",
    "META": "0001326801",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
}

NET_INCOME_CONCEPTS = ("NetIncomeLoss", "ProfitLoss")
OCF_CONCEPTS = (
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)
CAPEX_CONCEPTS = (
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsForAdditionsToPropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
)

OUTPUT_COLUMNS = [
    "observation_date",
    "release_timestamp",
    "effective_trade_date",
    "asset",
    "metric",
    "value",
    "score",
    "source",
    "is_proxy",
    "fetched_at",
    "fundamental_filed_through",
    "source_id",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("sec_valuation_collector")


def build_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": SEC_USER_AGENT,
            "Accept": "application/json,text/plain,*/*",
        }
    )
    return session


SESSION = build_session()


def validate_environment() -> None:
    if not SEC_USER_AGENT:
        raise RuntimeError(
            "SEC_USER_AGENT is empty. Add a GitHub Actions repository variable, "
            "for example: Mick investment-os your-email@example.com"
        )
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY is empty.")


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def next_nyse_session(timestamp: pd.Timestamp) -> str:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    local_date = ts.tz_convert("America/New_York").date()
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(
        start_date=(pd.Timestamp(local_date) + pd.Timedelta(days=1)).date(),
        end_date=(pd.Timestamp(local_date) + pd.Timedelta(days=14)).date(),
    )
    if schedule.empty:
        raise RuntimeError("Unable to resolve the next NYSE session.")
    return pd.Timestamp(schedule.index[0]).date().isoformat()


def logistic_score(
    value: float,
    midpoint: float,
    scale: float,
    higher_is_better: bool,
) -> float:
    direction = 1.0 if higher_is_better else -1.0
    z = direction * (float(value) - midpoint) / max(float(scale), 1e-9)
    z = float(np.clip(z, -20.0, 20.0))
    return float(np.clip(100.0 / (1.0 + math.exp(-z)), 0.0, 100.0))


def sec_get_json(url: str) -> dict[str, Any]:
    response = SESSION.get(url, timeout=(10, 60))
    response.raise_for_status()
    time.sleep(0.15)
    return response.json()


def fetch_company_facts(cik: str) -> dict[str, Any]:
    return sec_get_json(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    )


def concept_rows(
    company_facts: dict[str, Any],
    concept_names: Iterable[str],
) -> pd.DataFrame:
    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
    records: list[dict[str, Any]] = []

    for concept_name in concept_names:
        concept = us_gaap.get(concept_name)
        if not concept:
            continue
        for row in concept.get("units", {}).get("USD", []):
            item = dict(row)
            item["concept"] = concept_name
            records.append(item)

    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame(records)
    for column in ("start", "end", "filed"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["val"] = pd.to_numeric(frame.get("val"), errors="coerce")
    return frame.dropna(subset=["val", "filed", "end"])


def latest_annual_fact(
    company_facts: dict[str, Any],
    concept_names: Iterable[str],
) -> dict[str, Any] | None:
    frame = concept_rows(company_facts, concept_names)
    if frame.empty:
        return None

    if "form" in frame.columns:
        frame = frame[frame["form"].astype(str).isin(("10-K", "20-F", "40-F"))]
    if frame.empty:
        return None

    if "start" in frame.columns:
        duration_days = (frame["end"] - frame["start"]).dt.days
        frame = frame[duration_days.between(300, 430, inclusive="both")]
    if frame.empty:
        return None

    if "fp" in frame.columns:
        fy_rows = frame[frame["fp"].astype(str).str.upper().eq("FY")]
        if not fy_rows.empty:
            frame = fy_rows

    row = (
        frame.sort_values(["end", "filed"])
        .drop_duplicates(subset=["end"], keep="last")
        .iloc[-1]
    )
    return {
        "value": float(row["val"]),
        "filed": pd.Timestamp(row["filed"]),
        "period_end": pd.Timestamp(row["end"]),
        "accession": str(row.get("accn", "")),
        "concept": str(row.get("concept", "")),
    }


def latest_market_cap(ticker: str) -> tuple[float, str]:
    security = yf.Ticker(ticker)
    market_cap: float | None = None

    try:
        market_cap = float(security.fast_info["market_cap"])
    except Exception:
        try:
            market_cap = float(security.info["marketCap"])
        except Exception:
            market_cap = None

    history = security.history(period="10d", interval="1d", auto_adjust=False)
    if history.empty or "Close" not in history.columns:
        raise RuntimeError(f"No recent price history for {ticker}")
    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if close.empty:
        raise RuntimeError(f"No usable close price for {ticker}")

    price_date = pd.Timestamp(close.index[-1])
    if price_date.tzinfo is not None:
        price_date = price_date.tz_localize(None)

    if market_cap is None or not np.isfinite(market_cap) or market_cap <= 0:
        shares = security.get_shares_full(
            start=(pd.Timestamp.today() - pd.Timedelta(days=730)).date().isoformat()
        )
        if shares is None or shares.dropna().empty:
            raise RuntimeError(f"No market cap or share history for {ticker}")
        market_cap = float(shares.dropna().iloc[-1]) * float(close.iloc[-1])

    return market_cap, price_date.date().isoformat()


def latest_fred_value(series_id: str) -> tuple[float, str]:
    response = SESSION.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": "2020-01-01",
        },
        timeout=(10, 60),
    )
    response.raise_for_status()
    frame = pd.DataFrame(response.json().get("observations", []))
    if frame.empty:
        raise RuntimeError(f"FRED returned no data for {series_id}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"]).sort_values("date")
    if frame.empty:
        raise RuntimeError(f"FRED returned no numeric data for {series_id}")

    latest = frame.iloc[-1]
    return float(latest["value"]), latest["date"].date().isoformat()


def append_rows(new_rows: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
        output = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    else:
        output = new_rows.copy()

    for column in OUTPUT_COLUMNS:
        if column not in output.columns:
            output[column] = np.nan

    output = (
        output[OUTPUT_COLUMNS]
        .drop_duplicates(
            subset=["observation_date", "asset", "metric"],
            keep="last",
        )
        .sort_values(["effective_trade_date", "asset", "metric"])
    )
    output.to_csv(OUTPUT_FILE, index=False)


def collect_sec_valuation() -> None:
    LOGGER.info("Collector started; cwd=%s; output=%s", Path.cwd(), OUTPUT_FILE)
    validate_environment()
    fetched_at = utc_now()
    effective_trade_date = next_nyse_session(fetched_at)

    total_market_cap = 0.0
    total_net_income = 0.0
    total_ocf = 0.0
    total_capex = 0.0
    price_dates: list[str] = []
    filing_dates: list[pd.Timestamp] = []
    accessions: list[str] = []
    included: list[str] = []

    for ticker, cik in COMPANIES.items():
        try:
            LOGGER.info("Collecting SEC facts for %s", ticker)
            facts = fetch_company_facts(cik)
            net_income = latest_annual_fact(facts, NET_INCOME_CONCEPTS)
            ocf = latest_annual_fact(facts, OCF_CONCEPTS)
            capex = latest_annual_fact(facts, CAPEX_CONCEPTS)

            if not net_income or not ocf or not capex:
                LOGGER.warning(
                    "Skipping %s: incomplete standardized annual facts",
                    ticker,
                )
                continue

            market_cap, price_date = latest_market_cap(ticker)
            total_market_cap += market_cap
            total_net_income += net_income["value"]
            total_ocf += ocf["value"]
            total_capex += abs(capex["value"])
            price_dates.append(price_date)
            filing_dates.extend(
                [net_income["filed"], ocf["filed"], capex["filed"]]
            )
            accessions.extend(
                value
                for value in (
                    net_income["accession"],
                    ocf["accession"],
                    capex["accession"],
                )
                if value
            )
            included.append(ticker)
            LOGGER.info("%s valuation inputs accepted", ticker)
        except Exception:
            LOGGER.exception("Skipping %s because collection failed", ticker)

    if len(included) < 3:
        raise RuntimeError("Fewer than three companies had complete SEC inputs")
    if total_market_cap <= 0 or total_net_income <= 0:
        raise RuntimeError("Aggregate market cap or net income is invalid")

    free_cash_flow = total_ocf - total_capex
    trailing_pe = total_market_cap / total_net_income
    earnings_yield = 100.0 * total_net_income / total_market_cap
    fcf_yield = 100.0 * free_cash_flow / total_market_cap
    real_yield, real_yield_date = latest_fred_value("DFII10")
    erp = earnings_yield - real_yield

    observation_date = max(price_dates)
    fundamental_filed_through = max(filing_dates).date().isoformat()
    source_id = (
        f"companies={','.join(included)};"
        f"accessions={','.join(sorted(set(accessions)))};"
        f"DFII10_date={real_yield_date}"
    )

    metrics = [
        ("trailing_pe", trailing_pe, logistic_score(trailing_pe, 30.0, 6.0, False)),
        ("earnings_yield", earnings_yield, logistic_score(earnings_yield, 3.5, 0.8, True)),
        ("fcf_yield", fcf_yield, logistic_score(fcf_yield, 3.0, 1.0, True)),
        ("erp", erp, logistic_score(erp, 1.5, 0.8, True)),
    ]

    rows = pd.DataFrame(
        [
            {
                "observation_date": observation_date,
                "release_timestamp": fetched_at.isoformat(),
                "effective_trade_date": effective_trade_date,
                "asset": "AI_MEGA_CAP_BASKET",
                "metric": metric,
                "value": round(float(value), 6),
                "score": round(float(score), 4),
                "source": "SEC_XBRL_PLUS_YFINANCE_PLUS_FRED",
                "is_proxy": True,
                "fetched_at": fetched_at.isoformat(),
                "fundamental_filed_through": fundamental_filed_through,
                "source_id": source_id,
            }
            for metric, value, score in metrics
        ]
    )
    append_rows(rows)

    if not OUTPUT_FILE.is_file() or OUTPUT_FILE.stat().st_size == 0:
        raise RuntimeError(f"Output file was not written: {OUTPUT_FILE}")

    LOGGER.info(
        "Saved valuation proxy: companies=%s date=%s PE=%.2f EY=%.2f%% FCFY=%.2f%% ERP=%.2f%%",
        ",".join(included),
        observation_date,
        trailing_pe,
        earnings_yield,
        fcf_yield,
        erp,
    )
    LOGGER.info("Output: %s", OUTPUT_FILE)


def main() -> None:
    collect_sec_valuation()


if __name__ == "__main__":
    main()
