from __future__ import annotations

"""Collect free live-public PIT inputs for Investment OS.

Outputs:
- data/valuation_pit.csv
- data/macro_nowcast_pit.csv
- data/positioning_pit.csv
- data/events.csv
- data/ai_cycle_pit.csv

No paid data is used. Values that rely on a conservative publication-time
assumption or a derived basket are explicitly marked is_proxy=true.
"""

import logging
import math
import os
import re
import time
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

VALUATION_FILE = DATA_DIR / "valuation_pit.csv"
NOWCAST_FILE = DATA_DIR / "macro_nowcast_pit.csv"
POSITIONING_FILE = DATA_DIR / "positioning_pit.csv"
EVENTS_FILE = DATA_DIR / "events.csv"
AI_CYCLE_FILE = DATA_DIR / "ai_cycle_pit.csv"

FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "").strip()
CFTC_APP_TOKEN = os.getenv("CFTC_APP_TOKEN", "").strip()

COMPANIES: dict[str, str] = {
    "NVDA": "0001045810",
    "MSFT": "0000789019",
    "META": "0001326801",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
}

NET_INCOME_CONCEPTS = ("NetIncomeLoss", "ProfitLoss")
REVENUE_CONCEPTS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
)
OCF_CONCEPTS = (
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)
CAPEX_CONCEPTS = (
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsForAdditionsToPropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
)

VALUATION_COLUMNS = [
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
LOGGER = logging.getLogger("free_public_collectors")


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
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": SEC_USER_AGENT or "Investment-OS/1.0",
            "Accept": "application/json,text/csv,text/html,text/calendar,*/*",
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


def resolve_effective_trade_date(fetched_at: pd.Timestamp) -> str:
    """Resolve the first NYSE session on which newly fetched data may be used."""
    timestamp = pd.Timestamp(fetched_at)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    eastern = timestamp.tz_convert("America/New_York")
    local_date = pd.Timestamp(eastern.date())
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(
        start_date=(local_date - pd.Timedelta(days=1)).date(),
        end_date=(local_date + pd.Timedelta(days=14)).date(),
    )

    if local_date in schedule.index:
        market_open = pd.Timestamp(schedule.loc[local_date, "market_open"])
        if timestamp <= market_open:
            return local_date.date().isoformat()

    future = schedule[schedule["market_open"] > timestamp]
    if future.empty:
        raise RuntimeError("Unable to resolve effective NYSE trade date.")
    return pd.Timestamp(future.index[0]).date().isoformat()


def latest_completed_nyse_session(fetched_at: pd.Timestamp) -> str:
    """Return the latest NYSE session whose regular close has already passed."""
    timestamp = pd.Timestamp(fetched_at)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    eastern_date = pd.Timestamp(timestamp.tz_convert("America/New_York").date())
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(
        start_date=(eastern_date - pd.Timedelta(days=14)).date(),
        end_date=eastern_date.date(),
    )
    completed = schedule[schedule["market_close"] <= timestamp]
    if completed.empty:
        raise RuntimeError("Unable to resolve latest completed NYSE session.")
    return pd.Timestamp(completed.index[-1]).date().isoformat()


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


def append_pit_csv(
    path: Path,
    new_rows: pd.DataFrame,
    dedupe_keys: Iterable[str],
    sort_columns: Iterable[str] = ("effective_trade_date", "metric"),
) -> None:
    if new_rows.empty:
        LOGGER.warning("No rows to append to %s", path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        output = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    else:
        output = new_rows.copy()

    keys = [key for key in dedupe_keys if key in output.columns]
    if keys:
        output = output.drop_duplicates(subset=keys, keep="last")

    sort_keys = [key for key in sort_columns if key in output.columns]
    if sort_keys:
        output = output.sort_values(sort_keys)

    output.to_csv(path, index=False)
    LOGGER.info("Saved %s rows to %s", len(output), path)


def run_safely(name: str, function: Callable[[], None]) -> bool:
    try:
        function()
        LOGGER.info("%s completed", name)
        return True
    except Exception:
        LOGGER.exception("%s failed", name)
        return False


# ============================================================
# SEC helpers
# ============================================================

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
    return frame.dropna(subset=["val", "filed", "end"]).sort_values(
        ["end", "filed"]
    )


def annual_fact_history(
    company_facts: dict[str, Any],
    concept_names: Iterable[str],
) -> pd.DataFrame:
    frame = concept_rows(company_facts, concept_names)
    if frame.empty:
        return frame

    if "form" in frame.columns:
        frame = frame[frame["form"].astype(str).isin(("10-K", "20-F", "40-F"))]
    if "start" in frame.columns:
        duration = (frame["end"] - frame["start"]).dt.days
        frame = frame[duration.between(300, 430, inclusive="both")]
    if "fp" in frame.columns:
        fy = frame[frame["fp"].astype(str).str.upper().eq("FY")]
        if not fy.empty:
            frame = fy

    return (
        frame.sort_values(["end", "filed"])
        .drop_duplicates(subset=["end"], keep="last")
    )


def latest_annual_fact(
    company_facts: dict[str, Any],
    concept_names: Iterable[str],
) -> Optional[dict[str, Any]]:
    frame = annual_fact_history(company_facts, concept_names)
    if frame.empty:
        return None
    row = frame.iloc[-1]
    return {
        "value": float(row["val"]),
        "filed": pd.Timestamp(row["filed"]),
        "period_end": pd.Timestamp(row["end"]),
        "accession": str(row.get("accn", "")),
        "concept": str(row.get("concept", "")),
    }


def latest_period_yoy(
    company_facts: dict[str, Any],
    concept_names: Iterable[str],
    min_duration_days: int,
    max_duration_days: int,
) -> Optional[dict[str, Any]]:
    """Find the newest duration fact and a comparable prior-year duration fact."""
    frame = concept_rows(company_facts, concept_names)
    if frame.empty or "start" not in frame.columns:
        return None

    if "form" in frame.columns:
        frame = frame[frame["form"].astype(str).isin(("10-Q", "10-K"))]

    frame = frame.copy()
    frame["duration_days"] = (frame["end"] - frame["start"]).dt.days
    frame = frame[
        frame["duration_days"].between(
            min_duration_days,
            max_duration_days,
            inclusive="both",
        )
    ]
    frame = (
        frame.sort_values(["end", "filed"])
        .drop_duplicates(subset=["end", "duration_days"], keep="last")
    )

    if frame.empty:
        return None

    current = frame.iloc[-1]
    prior = frame[
        (frame["end"] >= current["end"] - pd.Timedelta(days=400))
        & (frame["end"] <= current["end"] - pd.Timedelta(days=330))
        & (
            (frame["duration_days"] - current["duration_days"]).abs()
            <= 20
        )
    ]
    if prior.empty:
        return None

    previous = prior.iloc[-1]
    previous_value = float(previous["val"])
    if previous_value == 0:
        return None

    return {
        "value": 100.0 * (float(current["val"]) / previous_value - 1.0),
        "filed": pd.Timestamp(current["filed"]),
        "period_end": pd.Timestamp(current["end"]),
        "accession": str(current.get("accn", "")),
        "concept": str(current.get("concept", "")),
    }


def latest_market_cap(ticker: str) -> tuple[float, str]:
    security = yf.Ticker(ticker)
    market_cap: Optional[float] = None

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
            "observation_start": "2012-01-01",
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


# ============================================================
# 1. SEC valuation proxy
# ============================================================

def collect_sec_valuation() -> None:
    fetched_at = utc_now()
    effective_trade_date = resolve_effective_trade_date(fetched_at)

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
            LOGGER.info("Collecting SEC valuation facts for %s", ticker)
            facts = fetch_company_facts(cik)
            net_income = latest_annual_fact(facts, NET_INCOME_CONCEPTS)
            ocf = latest_annual_fact(facts, OCF_CONCEPTS)
            capex = latest_annual_fact(facts, CAPEX_CONCEPTS)

            if not net_income or not ocf or not capex:
                LOGGER.warning("Skipping %s: incomplete annual facts", ticker)
                continue

            market_cap, price_date = latest_market_cap(ticker)
            total_market_cap += market_cap
            total_net_income += net_income["value"]
            total_ocf += ocf["value"]
            total_capex += abs(capex["value"])
            price_dates.append(price_date)
            filing_dates.extend([net_income["filed"], ocf["filed"], capex["filed"]])
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
        except Exception:
            LOGGER.exception("Skipping %s valuation inputs", ticker)

    if len(included) < 3:
        raise RuntimeError("Fewer than three companies had complete valuation inputs")
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

    append_pit_csv(
        VALUATION_FILE,
        rows,
        ("observation_date", "asset", "metric"),
        ("effective_trade_date", "asset", "metric"),
    )


# ============================================================
# 2. GDPNow
# ============================================================

def collect_gdpnow() -> None:
    response = SESSION.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": "GDPNOW",
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": "2012-01-01",
        },
        timeout=(10, 60),
    )
    response.raise_for_status()

    frame = pd.DataFrame(response.json().get("observations", []))
    if frame.empty:
        raise RuntimeError("FRED GDPNOW returned no rows")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"]).sort_values("date")
    if frame.empty:
        raise RuntimeError("FRED GDPNOW returned no numeric rows")

    latest = frame.iloc[-1]
    fetched_at = utc_now()
    value = float(latest["value"])
    score = logistic_score(value, midpoint=1.5, scale=1.25, higher_is_better=True)

    rows = pd.DataFrame(
        [
            {
                "observation_date": latest["date"].date().isoformat(),
                "release_timestamp": fetched_at.isoformat(),
                "effective_trade_date": resolve_effective_trade_date(fetched_at),
                "metric": "gdpnow_real_gdp_saar",
                "value": round(value, 4),
                "score": round(score, 4),
                "source": "FRED:GDPNOW",
                "is_proxy": True,
                "fetched_at": fetched_at.isoformat(),
            }
        ]
    )

    append_pit_csv(
        NOWCAST_FILE,
        rows,
        ("effective_trade_date", "metric", "source"),
    )


# ============================================================
# 3. Cboe Equity Put/Call
# ============================================================

def parse_cboe_daily_equity_put_call(html: str) -> float:
    """
    Parse the current Equity Put/Call Ratio from Cboe's official
    Daily Market Statistics page.

    The older equitypc.csv file is an archive and may stop years in
    the past, so it must not be treated as the latest daily value.
    """

    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        tables = []

    for table in tables:
        if table.empty:
            continue

        for _, row in table.iterrows():
            cells = [
                str(value).strip()
                for value in row.tolist()
                if pd.notna(value)
            ]

            row_text = " ".join(cells)
            normalized = re.sub(
                r"\s+",
                " ",
                row_text,
            ).upper()

            if "EQUITY PUT/CALL RATIO" not in normalized:
                continue

            numeric_values: list[float] = []

            for cell in cells:
                match = re.fullmatch(
                    r"\s*([0-9]+(?:\.[0-9]+)?)\s*",
                    cell,
                )

                if match:
                    numeric_values.append(
                        float(match.group(1))
                    )

            if numeric_values:
                return numeric_values[-1]

    text = BeautifulSoup(
        html,
        "html.parser",
    ).get_text(
        " ",
        strip=True,
    )

    match = re.search(
        r"EQUITY\s+PUT/CALL\s+RATIO"
        r"[^0-9]{0,80}"
        r"([0-9]+(?:\.[0-9]+)?)",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        raise RuntimeError(
            "Unable to parse the current Cboe "
            "Equity Put/Call Ratio"
        )

    return float(match.group(1))


def remove_invalid_put_call_rows() -> None:
    """
    Remove rows where the effective date is implausibly far from the
    observation date. This cleans the prior archive-row bug, such as
    a 2019 observation incorrectly activated in 2026.
    """

    if not POSITIONING_FILE.exists():
        return

    frame = pd.read_csv(
        POSITIONING_FILE
    )

    required = {
        "observation_date",
        "effective_trade_date",
        "metric",
    }

    if not required.issubset(
        frame.columns
    ):
        return

    observation = pd.to_datetime(
        frame["observation_date"],
        errors="coerce",
    )

    effective = pd.to_datetime(
        frame["effective_trade_date"],
        errors="coerce",
    )

    age_days = (
        effective
        - observation
    ).dt.days

    is_put_call = (
        frame["metric"]
        .astype(str)
        .str.lower()
        .eq("equity_put_call")
    )

    invalid = (
        is_put_call
        & (
            age_days.isna()
            | age_days.lt(0)
            | age_days.gt(10)
        )
    )

    if invalid.any():
        LOGGER.warning(
            "Removing %s invalid Equity Put/Call rows",
            int(invalid.sum()),
        )

        frame.loc[~invalid].to_csv(
            POSITIONING_FILE,
            index=False,
        )


def collect_cboe_put_call() -> None:
    fetched_at = utc_now()

    response = SESSION.get(
        "https://www.cboe.com/markets/"
        "us/options/market-statistics/daily/",
        timeout=(10, 60),
    )

    response.raise_for_status()

    value = parse_cboe_daily_equity_put_call(
        response.text
    )

    if not np.isfinite(value) or not 0 < value < 10:
        raise RuntimeError(
            "Cboe Equity Put/Call Ratio is outside "
            f"a plausible range: {value}"
        )

    observation_date = (
        latest_completed_nyse_session(
            fetched_at
        )
    )

    score = logistic_score(
        value,
        midpoint=0.70,
        scale=0.12,
        higher_is_better=True,
    )

    rows = pd.DataFrame(
        [
            {
                "observation_date":
                    observation_date,
                "release_timestamp":
                    fetched_at.isoformat(),
                "effective_trade_date":
                    resolve_effective_trade_date(
                        fetched_at
                    ),
                "metric":
                    "equity_put_call",
                "value":
                    round(value, 4),
                "score":
                    round(score, 4),
                "source":
                    "CBOE_DAILY_MARKET_STATISTICS",
                "is_proxy":
                    True,
                "fetched_at":
                    fetched_at.isoformat(),
            }
        ]
    )

    remove_invalid_put_call_rows()

    append_pit_csv(
        POSITIONING_FILE,
        rows,
        (
            "effective_trade_date",
            "metric",
            "source",
        ),
    )


# ============================================================
# 4. CFTC TFF positioning
# ============================================================

def collect_cftc_positioning() -> None:
    headers: dict[str, str] = {}
    if CFTC_APP_TOKEN:
        headers["X-App-Token"] = CFTC_APP_TOKEN

    response = SESSION.get(
        "https://publicreporting.cftc.gov/resource/gpe5-46if.json",
        params={
            "$limit": 5000,
            "$order": "report_date_as_yyyy_mm_dd ASC",
            "$where": (
                "report_date_as_yyyy_mm_dd >= '2012-01-01T00:00:00.000' "
                "AND upper(market_and_exchange_names) like '%NASDAQ-100%'"
            ),
        },
        headers=headers,
        timeout=(10, 90),
    )
    response.raise_for_status()
    records = response.json()
    if not records:
        raise RuntimeError("CFTC TFF API returned no NASDAQ-100 rows")

    frame = pd.DataFrame(records)
    required = {
        "report_date_as_yyyy_mm_dd",
        "open_interest_all",
        "lev_money_positions_long_all",
        "lev_money_positions_short_all",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"CFTC TFF response missing columns: {sorted(missing)}")

    frame["report_date"] = pd.to_datetime(
        frame["report_date_as_yyyy_mm_dd"],
        errors="coerce",
    )
    for column in (
        "open_interest_all",
        "lev_money_positions_long_all",
        "lev_money_positions_short_all",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(
        subset=[
            "report_date",
            "open_interest_all",
            "lev_money_positions_long_all",
            "lev_money_positions_short_all",
        ]
    ).sort_values("report_date")
    frame = frame[frame["open_interest_all"] > 0]
    if frame.empty:
        raise RuntimeError("CFTC TFF response contains no usable rows")

    frame["net_pct_open_interest"] = (
        100.0
        * (
            frame["lev_money_positions_long_all"]
            - frame["lev_money_positions_short_all"]
        )
        / frame["open_interest_all"]
    )

    current = frame.iloc[-1]
    trailing = frame.tail(156)["net_pct_open_interest"].dropna()
    percentile = 100.0 * (
        (trailing < float(current["net_pct_open_interest"])).sum()
        + 0.5 * (trailing == float(current["net_pct_open_interest"])).sum()
    ) / len(trailing)
    score = float(np.clip(100.0 - percentile, 0.0, 100.0))

    report_date = pd.Timestamp(current["report_date"]).tz_localize(None)
    release_local = (
        report_date
        + pd.Timedelta(days=3)
        + pd.Timedelta(hours=15, minutes=30)
    ).tz_localize("America/New_York")
    release_timestamp = release_local.tz_convert("UTC")

    rows = pd.DataFrame(
        [
            {
                "observation_date": report_date.date().isoformat(),
                "release_timestamp": release_timestamp.isoformat(),
                "effective_trade_date": resolve_effective_trade_date(
                    release_timestamp
                ),
                "metric": "cftc_positioning",
                "value": round(float(current["net_pct_open_interest"]), 6),
                "score": round(score, 4),
                "source": "CFTC_TFF_FUTURES_ONLY_GPE5_46IF",
                "is_proxy": True,
                "fetched_at": utc_now().isoformat(),
            }
        ]
    )

    append_pit_csv(
        POSITIONING_FILE,
        rows,
        ("observation_date", "metric", "source"),
    )


# ============================================================
# 5. Events
# ============================================================

def unfold_ics(text: str) -> list[str]:
    output: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.startswith((" ", "\t")) and output:
            output[-1] += line[1:]
        else:
            output.append(line)
    return output


def parse_ics_events(text: str, source: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current: dict[str, str] = {}
    inside = False

    for line in unfold_ics(text):
        if line == "BEGIN:VEVENT":
            current = {}
            inside = True
            continue
        if line == "END:VEVENT":
            inside = False
            date_raw = current.get("DTSTART", "")
            summary = current.get("SUMMARY", "").replace("\\,", ",").strip()
            date_match = re.search(r"(\d{8})", date_raw)
            if date_match and summary:
                event_date = pd.to_datetime(
                    date_match.group(1),
                    format="%Y%m%d",
                    errors="coerce",
                )
                if pd.notna(event_date):
                    events.append(
                        {
                            "event_date": event_date.date().isoformat(),
                            "asset": "MARKET",
                            "event_type": "macro_release",
                            "description": summary,
                            "source": source,
                            "is_proxy": False,
                        }
                    )
            current = {}
            continue

        if inside and ":" in line:
            key, value = line.split(":", 1)
            current[key.split(";", 1)[0]] = value

    return events


def collect_bls_events() -> list[dict[str, Any]]:
    response = SESSION.get(
        "https://www.bls.gov/schedule/news_release/bls.ics",
        timeout=(10, 60),
    )
    response.raise_for_status()
    return parse_ics_events(response.text, "BLS_ICS")


def collect_bea_events() -> list[dict[str, Any]]:
    response = SESSION.get("https://www.bea.gov/news/schedule", timeout=(10, 60))
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    events: list[dict[str, Any]] = []

    for table in tables:
        if table.empty:
            continue
        columns = [str(column).strip().lower() for column in table.columns]
        table.columns = columns

        date_column = next(
            (column for column in columns if "date" in column),
            None,
        )
        release_column = next(
            (
                column
                for column in columns
                if "release" in column or "title" in column
            ),
            None,
        )
        if date_column is None:
            continue

        for _, row in table.iterrows():
            event_date = pd.to_datetime(row.get(date_column), errors="coerce")
            if pd.isna(event_date):
                continue
            description = str(
                row.get(release_column)
                if release_column is not None
                else "BEA economic release"
            ).strip()
            if not description or description.lower() == "nan":
                description = "BEA economic release"

            events.append(
                {
                    "event_date": event_date.date().isoformat(),
                    "asset": "MARKET",
                    "event_type": "macro_release",
                    "description": description,
                    "source": "BEA_RELEASE_SCHEDULE",
                    "is_proxy": False,
                }
            )

    return events


def collect_fomc_events() -> list[dict[str, Any]]:
    """
    Collect one policy-decision date per scheduled FOMC meeting.

    The decision date is the final day of each meeting range.
    Section boundaries are determined by every FOMC-year heading.
    """

    response = SESSION.get(
        "https://www.federalreserve.gov/"
        "monetarypolicy/fomccalendars.htm",
        timeout=(10, 60),
    )

    response.raise_for_status()

    text = BeautifulSoup(
        response.text,
        "html.parser",
    ).get_text(
        "\n",
        strip=True,
    )

    heading_pattern = re.compile(
        r"\b(?P<year>20\d{2})\s+FOMC\s+Meetings\b",
        flags=re.IGNORECASE,
    )

    heading_matches = list(
        heading_pattern.finditer(text)
    )

    if not heading_matches:
        raise RuntimeError(
            "Unable to locate FOMC year sections"
        )

    month_numbers = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }

    current_year = pd.Timestamp.today().year
    wanted_years = {
        current_year,
        current_year + 1,
    }

    events: list[dict[str, Any]] = []

    for index, heading in enumerate(
        heading_matches
    ):
        year = int(
            heading.group("year")
        )

        if year not in wanted_years:
            continue

        section_start = heading.end()
        section_end = (
            heading_matches[index + 1].start()
            if index + 1 < len(heading_matches)
            else len(text)
        )

        section = text[
            section_start:section_end
        ]

        lines = [
            line.strip()
            for line in section.splitlines()
            if line.strip()
        ]

        for line_index in range(
            len(lines) - 1
        ):
            month_name = lines[line_index]

            if month_name not in month_numbers:
                continue

            date_token = (
                lines[line_index + 1]
                .replace("*", "")
                .strip()
            )

            if not re.fullmatch(
                r"\d{1,2}(?:-\d{1,2})?",
                date_token,
            ):
                continue

            decision_day = int(
                date_token.split("-")[-1]
            )

            event_date = pd.Timestamp(
                year=year,
                month=month_numbers[
                    month_name
                ],
                day=decision_day,
            )

            events.append(
                {
                    "event_date":
                        event_date
                        .date()
                        .isoformat(),
                    "asset":
                        "MARKET",
                    "event_type":
                        "fomc",
                    "description":
                        "FOMC policy decision",
                    "source":
                        "FED_FOMC_CALENDAR",
                    "is_proxy":
                        False,
                }
            )

    output = (
        pd.DataFrame(events)
        .drop_duplicates(
            subset=[
                "event_date",
                "asset",
                "event_type",
                "description",
                "source",
            ],
            keep="last",
        )
        .sort_values("event_date")
    )

    if output.empty:
        raise RuntimeError(
            "No current or next-year FOMC dates were parsed"
        )

    return output.to_dict(
        "records"
    )


def extract_earnings_dates(calendar: Any) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    if isinstance(calendar, dict):
        raw = calendar.get("Earnings Date") or calendar.get("EarningsDate")
        if isinstance(raw, (list, tuple)):
            dates.extend(pd.Timestamp(value) for value in raw)
        elif raw is not None:
            dates.append(pd.Timestamp(raw))
    elif isinstance(calendar, pd.DataFrame) and not calendar.empty:
        for value in calendar.to_numpy().ravel():
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.notna(parsed):
                dates.append(pd.Timestamp(parsed))
    return dates


def collect_earnings_events() -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for ticker in COMPANIES:
        try:
            calendar = yf.Ticker(ticker).calendar
            dates = extract_earnings_dates(calendar)
            for date in dates:
                if date.tzinfo is not None:
                    date = date.tz_convert("America/New_York").tz_localize(None)
                events.append(
                    {
                        "event_date": date.date().isoformat(),
                        "asset": ticker,
                        "event_type": "earnings",
                        "description": f"{ticker} estimated earnings date",
                        "source": "YFINANCE_EARNINGS_PROXY",
                        "is_proxy": True,
                    }
                )
        except Exception:
            LOGGER.exception("Unable to collect earnings date for %s", ticker)
    return events


def collect_events() -> None:
    rows: list[dict[str, Any]] = []
    for name, function in (
        ("BLS", collect_bls_events),
        ("BEA", collect_bea_events),
        ("FOMC", collect_fomc_events),
        ("earnings", collect_earnings_events),
    ):
        try:
            rows.extend(function())
        except Exception:
            LOGGER.exception("%s event source failed", name)

    if not rows:
        raise RuntimeError("No event source returned usable rows")

    frame = pd.DataFrame(rows)
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    today = pd.Timestamp.today().normalize()
    frame = frame[
        frame["event_date"].between(
            today - pd.Timedelta(days=10),
            today + pd.Timedelta(days=400),
        )
    ].copy()
    frame["event_date"] = frame["event_date"].dt.date.astype(str)

    frame = (
        frame
        .drop_duplicates(
            subset=[
                "event_date",
                "asset",
                "event_type",
                "description",
                "source",
            ],
            keep="last",
        )
        .sort_values(
            [
                "event_date",
                "asset",
                "event_type",
            ]
        )
    )

    frame.to_csv(
        EVENTS_FILE,
        index=False,
    )

    LOGGER.info(
        "Saved %s current event rows to %s",
        len(frame),
        EVENTS_FILE,
    )


# ============================================================
# 6. AI Cycle proxies
# ============================================================

def conservative_release_timestamp(filed_date: pd.Timestamp) -> pd.Timestamp:
    """Use 17:00 ET on the filing date when SEC accepted time is unavailable."""
    local = (
        pd.Timestamp(filed_date)
        .tz_localize(None)
        .normalize()
        + pd.Timedelta(hours=17)
    ).tz_localize("America/New_York")
    return local.tz_convert("UTC")


def collect_ai_cycle() -> None:
    fetched_at = utc_now()
    rows: list[dict[str, Any]] = []

    nvda_facts = fetch_company_facts(COMPANIES["NVDA"])
    nvda_revenue = latest_period_yoy(
        nvda_facts,
        REVENUE_CONCEPTS,
        min_duration_days=70,
        max_duration_days=120,
    )
    if nvda_revenue:
        release = conservative_release_timestamp(nvda_revenue["filed"])
        value = float(nvda_revenue["value"])
        rows.append(
            {
                "observation_date": nvda_revenue["period_end"].date().isoformat(),
                "release_timestamp": release.isoformat(),
                "effective_trade_date": resolve_effective_trade_date(release),
                "company": "NVDA",
                "metric": "nvidia_revenue_yoy_proxy",
                "value": round(value, 6),
                "score": round(
                    logistic_score(value, midpoint=25.0, scale=15.0, higher_is_better=True),
                    4,
                ),
                "source": "SEC_XBRL_QUARTERLY_REVENUE",
                "is_proxy": True,
                "fetched_at": fetched_at.isoformat(),
                "source_id": nvda_revenue["accession"],
            }
        )

    capex_values: list[float] = []
    capex_filings: list[pd.Timestamp] = []
    capex_periods: list[pd.Timestamp] = []
    capex_accessions: list[str] = []

    for ticker in ("MSFT", "META", "GOOGL", "AMZN"):
        try:
            facts = fetch_company_facts(COMPANIES[ticker])
            result = latest_period_yoy(
                facts,
                CAPEX_CONCEPTS,
                min_duration_days=70,
                max_duration_days=300,
            )
            if result is None:
                LOGGER.warning("No comparable CapEx YoY fact for %s", ticker)
                continue
            capex_values.append(float(result["value"]))
            capex_filings.append(pd.Timestamp(result["filed"]))
            capex_periods.append(pd.Timestamp(result["period_end"]))
            if result["accession"]:
                capex_accessions.append(str(result["accession"]))
        except Exception:
            LOGGER.exception("Unable to collect CapEx YoY for %s", ticker)

    if capex_values:
        value = float(np.mean(capex_values))
        filed = max(capex_filings)
        release = conservative_release_timestamp(filed)
        rows.append(
            {
                "observation_date": max(capex_periods).date().isoformat(),
                "release_timestamp": release.isoformat(),
                "effective_trade_date": resolve_effective_trade_date(release),
                "company": "MSFT_META_GOOGL_AMZN",
                "metric": "hyperscaler_capex_yoy",
                "value": round(value, 6),
                "score": round(
                    logistic_score(value, midpoint=20.0, scale=12.0, higher_is_better=True),
                    4,
                ),
                "source": "SEC_XBRL_YTD_CAPEX_PROXY",
                "is_proxy": True,
                "fetched_at": fetched_at.isoformat(),
                "source_id": ",".join(sorted(set(capex_accessions))),
            }
        )

    if not rows:
        raise RuntimeError("No usable AI Cycle proxy rows were collected")

    append_pit_csv(
        AI_CYCLE_FILE,
        pd.DataFrame(rows),
        ("observation_date", "company", "metric"),
        ("effective_trade_date", "company", "metric"),
    )


def main() -> None:
    validate_environment()

    results = {
        "SEC Valuation": run_safely("SEC Valuation", collect_sec_valuation),
        "GDPNow": run_safely("GDPNow", collect_gdpnow),
        "Cboe Put/Call": run_safely("Cboe Put/Call", collect_cboe_put_call),
        "CFTC": run_safely("CFTC", collect_cftc_positioning),
        "Events": run_safely("Events", collect_events),
        "AI Cycle": run_safely("AI Cycle", collect_ai_cycle),
    }

    LOGGER.info("Collector summary: %s", results)

    # Valuation remains required because the existing production engine uses it
    # as the core free-data extension. Other sources are optional and are
    # reflected through data-quality coverage if temporarily unavailable.
    if not VALUATION_FILE.exists() or VALUATION_FILE.stat().st_size == 0:
        raise RuntimeError("Required valuation_pit.csv was not generated")


if __name__ == "__main__":
    main()
