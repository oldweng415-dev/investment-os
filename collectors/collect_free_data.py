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

import hashlib
import logging
import math
import os
import re
import time
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from pypdf import PdfReader
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
OPTIONS_UNDERLYING = os.getenv("OPTIONS_UNDERLYING", "QQQ").strip().upper()
OPTIONS_TARGET_DTE = int(os.getenv("OPTIONS_TARGET_DTE", "30"))

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



def comparable_duration_pair(
    company_facts: dict[str, Any],
    concept_names: Iterable[str],
    min_duration_days: int,
    max_duration_days: int,
) -> Optional[dict[str, Any]]:
    """Return the newest fact and a comparable prior-year duration fact."""
    frame = concept_rows(
        company_facts,
        concept_names,
    )

    if frame.empty or "start" not in frame.columns:
        return None

    if "form" in frame.columns:
        frame = frame[
            frame["form"]
            .astype(str)
            .isin(("10-Q", "10-K"))
        ]

    frame = frame.copy()
    frame["duration_days"] = (
        frame["end"]
        - frame["start"]
    ).dt.days

    frame = frame[
        frame["duration_days"].between(
            min_duration_days,
            max_duration_days,
            inclusive="both",
        )
    ]

    frame = (
        frame
        .sort_values(
            ["end", "filed"]
        )
        .drop_duplicates(
            subset=[
                "end",
                "duration_days",
            ],
            keep="last",
        )
    )

    if frame.empty:
        return None

    for current_index in range(
        len(frame) - 1,
        -1,
        -1,
    ):
        current = frame.iloc[
            current_index
        ]

        prior = frame[
            frame["end"].between(
                current["end"]
                - pd.Timedelta(days=400),
                current["end"]
                - pd.Timedelta(days=330),
            )
            & (
                frame["duration_days"]
                .sub(
                    current["duration_days"]
                )
                .abs()
                <= 20
            )
        ]

        if prior.empty:
            continue

        previous = prior.iloc[-1]
        previous_value = float(
            previous["val"]
        )

        if previous_value == 0:
            continue

        return {
            "current_value": float(
                current["val"]
            ),
            "previous_value":
                previous_value,
            "yoy": 100.0 * (
                float(current["val"])
                / previous_value
                - 1.0
            ),
            "filed": pd.Timestamp(
                current["filed"]
            ),
            "period_end": pd.Timestamp(
                current["end"]
            ),
            "duration_days": int(
                current["duration_days"]
            ),
            "accession": str(
                current.get(
                    "accn",
                    "",
                )
            ),
            "concept": str(
                current.get(
                    "concept",
                    "",
                )
            ),
        }

    return None


def latest_ttm_fact(
    company_facts: dict[str, Any],
    concept_names: Iterable[str],
) -> Optional[dict[str, Any]]:
    """
    Build a free SEC TTM value using:

        latest fiscal year
        + latest current-year YTD
        - comparable prior-year YTD

    If a usable YTD pair is unavailable, conservatively fall back to
    the latest annual filing and identify that basis explicitly.
    """

    annual = latest_annual_fact(
        company_facts,
        concept_names,
    )

    if annual is None:
        return None

    ytd = comparable_duration_pair(
        company_facts,
        concept_names,
        min_duration_days=150,
        max_duration_days=300,
    )

    if (
        ytd is not None
        and ytd["period_end"]
        > annual["period_end"]
    ):
        value = (
            float(annual["value"])
            + float(ytd["current_value"])
            - float(ytd["previous_value"])
        )

        return {
            "value": value,
            "filed": max(
                pd.Timestamp(
                    annual["filed"]
                ),
                pd.Timestamp(
                    ytd["filed"]
                ),
            ),
            "period_end":
                pd.Timestamp(
                    ytd["period_end"]
                ),
            "accession": ",".join(
                sorted(
                    {
                        str(
                            annual.get(
                                "accession",
                                "",
                            )
                        ),
                        str(
                            ytd.get(
                                "accession",
                                "",
                            )
                        ),
                    }
                    - {""}
                )
            ),
            "concept": str(
                annual.get(
                    "concept",
                    "",
                )
            ),
            "basis": "TTM",
        }

    return {
        **annual,
        "basis": "ANNUAL_FALLBACK",
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
    """
    Build an AI mega-cap basket valuation from SEC-derived TTM
    fundamentals and current public market capitalizations.

    True forward estimates are intentionally not fabricated.
    """

    fetched_at = utc_now()
    effective_trade_date = (
        resolve_effective_trade_date(
            fetched_at
        )
    )

    total_market_cap = 0.0
    total_net_income = 0.0
    total_ocf = 0.0
    total_capex = 0.0

    price_dates: list[str] = []
    filing_dates: list[pd.Timestamp] = []
    accessions: list[str] = []
    included: list[str] = []
    basis_by_company: list[str] = []

    for ticker, cik in COMPANIES.items():
        try:
            LOGGER.info(
                "Collecting SEC TTM valuation facts for %s",
                ticker,
            )

            facts = fetch_company_facts(
                cik
            )

            net_income = latest_ttm_fact(
                facts,
                NET_INCOME_CONCEPTS,
            )

            ocf = latest_ttm_fact(
                facts,
                OCF_CONCEPTS,
            )

            capex = latest_ttm_fact(
                facts,
                CAPEX_CONCEPTS,
            )

            if (
                net_income is None
                or ocf is None
                or capex is None
            ):
                LOGGER.warning(
                    "Skipping %s: incomplete TTM/annual facts",
                    ticker,
                )
                continue

            market_cap, price_date = (
                latest_market_cap(
                    ticker
                )
            )

            total_market_cap += (
                market_cap
            )

            total_net_income += float(
                net_income["value"]
            )

            total_ocf += float(
                ocf["value"]
            )

            total_capex += abs(
                float(
                    capex["value"]
                )
            )

            price_dates.append(
                price_date
            )

            filing_dates.extend(
                [
                    pd.Timestamp(
                        net_income["filed"]
                    ),
                    pd.Timestamp(
                        ocf["filed"]
                    ),
                    pd.Timestamp(
                        capex["filed"]
                    ),
                ]
            )

            for fact in (
                net_income,
                ocf,
                capex,
            ):
                accession = str(
                    fact.get(
                        "accession",
                        "",
                    )
                )

                if accession:
                    accessions.extend(
                        value
                        for value in accession.split(
                            ","
                        )
                        if value
                    )

            basis_by_company.append(
                f"{ticker}:"
                f"{net_income['basis']}/"
                f"{ocf['basis']}/"
                f"{capex['basis']}"
            )

            included.append(
                ticker
            )

        except Exception:
            LOGGER.exception(
                "Skipping %s valuation inputs",
                ticker,
            )

    if len(included) < 3:
        raise RuntimeError(
            "Fewer than three companies had "
            "complete TTM valuation inputs"
        )

    if (
        total_market_cap <= 0
        or total_net_income <= 0
    ):
        raise RuntimeError(
            "Aggregate market cap or TTM "
            "net income is invalid"
        )

    free_cash_flow = (
        total_ocf
        - total_capex
    )

    trailing_pe = (
        total_market_cap
        / total_net_income
    )

    earnings_yield = (
        100.0
        * total_net_income
        / total_market_cap
    )

    fcf_yield = (
        100.0
        * free_cash_flow
        / total_market_cap
    )

    real_yield, real_yield_date = (
        latest_fred_value(
            "DFII10"
        )
    )

    erp = (
        earnings_yield
        - real_yield
    )

    observation_date = max(
        price_dates
    )

    fundamental_filed_through = (
        max(
            filing_dates
        )
        .date()
        .isoformat()
    )

    source_id = (
        f"companies={','.join(included)};"
        f"basis={','.join(basis_by_company)};"
        f"accessions={','.join(sorted(set(accessions)))};"
        f"DFII10_date={real_yield_date}"
    )

    metrics = [
        (
            "trailing_pe",
            trailing_pe,
            logistic_score(
                trailing_pe,
                30.0,
                6.0,
                False,
            ),
        ),
        (
            "earnings_yield",
            earnings_yield,
            logistic_score(
                earnings_yield,
                3.5,
                0.8,
                True,
            ),
        ),
        (
            "fcf_yield",
            fcf_yield,
            logistic_score(
                fcf_yield,
                3.0,
                1.0,
                True,
            ),
        ),
        (
            "erp",
            erp,
            logistic_score(
                erp,
                1.5,
                0.8,
                True,
            ),
        ),
    ]

    rows = pd.DataFrame(
        [
            {
                "observation_date":
                    observation_date,
                "release_timestamp":
                    fetched_at.isoformat(),
                "effective_trade_date":
                    effective_trade_date,
                "asset":
                    "AI_MEGA_CAP_BASKET",
                "metric":
                    metric,
                "value":
                    round(
                        float(value),
                        6,
                    ),
                "score":
                    round(
                        float(score),
                        4,
                    ),
                "source":
                    "SEC_XBRL_TTM_PLUS_"
                    "YFINANCE_PLUS_FRED",
                "is_proxy":
                    True,
                "fetched_at":
                    fetched_at.isoformat(),
                "fundamental_filed_through":
                    fundamental_filed_through,
                "fundamental_basis":
                    "TTM_WITH_ANNUAL_FALLBACK",
                "constituent_count":
                    len(included),
                "source_id":
                    source_id,
            }
            for metric, value, score
            in metrics
        ]
    )

    append_pit_csv(
        VALUATION_FILE,
        rows,
        (
            "observation_date",
            "asset",
            "metric",
        ),
        (
            "effective_trade_date",
            "asset",
            "metric",
        ),
    )

# ============================================================
# 2. GDPNow
# ============================================================


def collect_gdpnow() -> None:
    """
    Preserve GDPNow vintages only when the observed value changes.

    FRED exposes the current public history, not a complete immutable
    vintage history. The first time Investment OS observes a changed
    value, fetched_at becomes the conservative public-release proxy.
    """

    response = SESSION.get(
        "https://api.stlouisfed.org/"
        "fred/series/observations",
        params={
            "series_id":
                "GDPNOW",
            "api_key":
                FRED_API_KEY,
            "file_type":
                "json",
            "observation_start":
                "2012-01-01",
        },
        timeout=(10, 60),
    )

    response.raise_for_status()

    frame = pd.DataFrame(
        response.json().get(
            "observations",
            [],
        )
    )

    if frame.empty:
        raise RuntimeError(
            "FRED GDPNOW returned no rows"
        )

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    frame["value"] = pd.to_numeric(
        frame["value"],
        errors="coerce",
    )

    frame = (
        frame
        .dropna(
            subset=[
                "date",
                "value",
            ]
        )
        .sort_values(
            "date"
        )
    )

    if frame.empty:
        raise RuntimeError(
            "FRED GDPNOW returned "
            "no numeric rows"
        )

    latest = frame.iloc[-1]
    fetched_at = utc_now()
    observation_date = (
        latest["date"]
        .date()
        .isoformat()
    )
    value = float(
        latest["value"]
    )

    if NOWCAST_FILE.exists():
        existing = pd.read_csv(
            NOWCAST_FILE
        )

        same_metric = existing[
            existing["metric"]
            .astype(str)
            .str.lower()
            .eq(
                "gdpnow_real_gdp_saar"
            )
        ].copy()

        if not same_metric.empty:
            same_metric[
                "release_timestamp"
            ] = pd.to_datetime(
                same_metric[
                    "release_timestamp"
                ],
                errors="coerce",
                utc=True,
            )

            same_metric = (
                same_metric
                .sort_values(
                    "release_timestamp"
                )
            )

            last_row = (
                same_metric.iloc[-1]
            )

            last_value = pd.to_numeric(
                last_row.get(
                    "value"
                ),
                errors="coerce",
            )

            last_observation = str(
                last_row.get(
                    "observation_date",
                    "",
                )
            )

            if (
                pd.notna(
                    last_value
                )
                and math.isclose(
                    float(last_value),
                    value,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and last_observation
                == observation_date
            ):
                LOGGER.info(
                    "GDPNow unchanged at %.4f; "
                    "no duplicate vintage written",
                    value,
                )
                return

    score = logistic_score(
        value,
        midpoint=1.5,
        scale=1.25,
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
                    "gdpnow_real_gdp_saar",
                "value":
                    round(
                        value,
                        4,
                    ),
                "score":
                    round(
                        score,
                        4,
                    ),
                "source":
                    "FRED:GDPNOW",
                "is_proxy":
                    True,
                "fetched_at":
                    fetched_at.isoformat(),
                "vintage_method":
                    "FIRST_OBSERVED_VALUE_CHANGE",
            }
        ]
    )

    append_pit_csv(
        NOWCAST_FILE,
        rows,
        (
            "release_timestamp",
            "metric",
            "source",
        ),
        (
            "effective_trade_date",
            "release_timestamp",
        ),
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
    """
    Collect Nasdaq-100 leveraged-money positioning from the official
    CFTC TFF Futures Only dataset.

    The API query intentionally avoids a fragile server-side market
    name filter. Recent rows are fetched first, then Nasdaq-100
    contracts are selected and aggregated locally.
    """

    headers: dict[str, str] = {}

    if CFTC_APP_TOKEN:
        headers[
            "X-App-Token"
        ] = CFTC_APP_TOKEN

    fetched_at = utc_now()
    start_date = (
        fetched_at
        .tz_convert(
            "America/New_York"
        )
        .normalize()
        - pd.Timedelta(
            days=365 * 5
        )
    )

    response = SESSION.get(
        "https://publicreporting.cftc.gov/"
        "resource/gpe5-46if.json",
        params={
            "$select": ",".join(
                [
                    "report_date_as_yyyy_mm_dd",
                    "market_and_exchange_names",
                    "contract_market_name",
                    "cftc_contract_market_code",
                    "open_interest_all",
                    "lev_money_positions_long",
                    "lev_money_positions_short",
                ]
            ),
            "$limit":
                50000,
            "$order":
                "report_date_as_yyyy_mm_dd ASC",
            "$where":
                "report_date_as_yyyy_mm_dd >= "
                f"'{start_date.date().isoformat()}T00:00:00.000'",
        },
        headers=headers,
        timeout=(10, 120),
    )

    if not response.ok:
        raise RuntimeError(
            "CFTC API request failed: "
            f"status={response.status_code}; "
            f"body={response.text[:1000]}"
        )

    records = response.json()

    if not records:
        raise RuntimeError(
            "CFTC TFF API returned no recent rows"
        )

    frame = pd.DataFrame(
        records
    )

    required = {
        "report_date_as_yyyy_mm_dd",
        "open_interest_all",
        "lev_money_positions_long",
        "lev_money_positions_short",
    }

    missing = (
        required
        - set(
            frame.columns
        )
    )

    if missing:
        raise RuntimeError(
            "CFTC TFF response missing columns: "
            f"{sorted(missing)}"
        )

    market_text = pd.Series(
        "",
        index=frame.index,
        dtype=str,
    )

    for column in (
        "market_and_exchange_names",
        "contract_market_name",
    ):
        if column in frame.columns:
            market_text = (
                market_text
                + " "
                + frame[column]
                .fillna("")
                .astype(str)
            )

    nasdaq_mask = (
        market_text
        .str.upper()
        .str.contains(
            r"NASDAQ[\s\-]*100"
            r"|NASDAQ\s+100"
            r"|E[\-\s]*MINI\s+NASDAQ",
            regex=True,
            na=False,
        )
    )

    frame = frame.loc[
        nasdaq_mask
    ].copy()

    if frame.empty:
        examples = (
            market_text
            .drop_duplicates()
            .head(20)
            .tolist()
        )

        raise RuntimeError(
            "CFTC TFF API returned no locally "
            "matched Nasdaq-100 rows; "
            f"sample markets={examples}"
        )

    frame["report_date"] = pd.to_datetime(
        frame[
            "report_date_as_yyyy_mm_dd"
        ],
        errors="coerce",
    )

    numeric_columns = (
        "open_interest_all",
        "lev_money_positions_long",
        "lev_money_positions_short",
    )

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.dropna(
        subset=[
            "report_date",
            *numeric_columns,
        ]
    )

    frame = frame[
        frame[
            "open_interest_all"
        ] > 0
    ]

    if frame.empty:
        raise RuntimeError(
            "CFTC Nasdaq-100 rows contain "
            "no usable numeric positions"
        )

    # Aggregate E-mini/Micro or multiple Nasdaq-100 contract rows.
    grouped = (
        frame.groupby(
            "report_date",
            as_index=False,
        )[
            list(
                numeric_columns
            )
        ]
        .sum()
        .sort_values(
            "report_date"
        )
    )

    grouped[
        "net_pct_open_interest"
    ] = (
        100.0
        * (
            grouped[
                "lev_money_positions_long"
            ]
            - grouped[
                "lev_money_positions_short"
            ]
        )
        / grouped[
            "open_interest_all"
        ]
    )

    # COT observation is Tuesday; public release is normally Friday.
    grouped[
        "release_timestamp"
    ] = grouped[
        "report_date"
    ].map(
        lambda report_date: (
            pd.Timestamp(
                report_date
            )
            + pd.Timedelta(
                days=3
            )
            + pd.Timedelta(
                hours=15,
                minutes=30,
            )
        )
        .tz_localize(
            "America/New_York"
        )
        .tz_convert(
            "UTC"
        )
    )

    available = grouped[
        grouped[
            "release_timestamp"
        ] <= fetched_at
    ].copy()

    if available.empty:
        raise RuntimeError(
            "CFTC rows exist but none have "
            "a release time available yet"
        )

    current = available.iloc[-1]
    trailing = (
        available
        .tail(156)[
            "net_pct_open_interest"
        ]
        .dropna()
    )

    if len(trailing) < 52:
        raise RuntimeError(
            "CFTC Nasdaq-100 history has fewer "
            "than 52 usable weekly rows"
        )

    current_value = float(
        current[
            "net_pct_open_interest"
        ]
    )

    percentile = (
        100.0
        * (
            (
                trailing
                < current_value
            ).sum()
            + 0.5
            * (
                trailing
                == current_value
            ).sum()
        )
        / len(
            trailing
        )
    )

    score = float(
        np.clip(
            100.0
            - percentile,
            0.0,
            100.0,
        )
    )

    report_date = pd.Timestamp(
        current[
            "report_date"
        ]
    ).tz_localize(None)

    release_timestamp = pd.Timestamp(
        current[
            "release_timestamp"
        ]
    )

    rows = pd.DataFrame(
        [
            {
                "observation_date":
                    report_date
                    .date()
                    .isoformat(),
                "release_timestamp":
                    release_timestamp
                    .isoformat(),
                "effective_trade_date":
                    resolve_effective_trade_date(
                        release_timestamp
                    ),
                "metric":
                    "cftc_positioning",
                "value":
                    round(
                        current_value,
                        6,
                    ),
                "score":
                    round(
                        score,
                        4,
                    ),
                "source":
                    "CFTC_TFF_FUTURES_ONLY_"
                    "NASDAQ100_AGGREGATED",
                "is_proxy":
                    True,
                "fetched_at":
                    fetched_at.isoformat(),
                "contract_rows":
                    int(
                        frame[
                            frame[
                                "report_date"
                            ].eq(
                                report_date
                            )
                        ].shape[0]
                    ),
            }
        ]
    )

    append_pit_csv(
        POSITIONING_FILE,
        rows,
        (
            "observation_date",
            "metric",
            "source",
        ),
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
    """
    Collect major BEA event dates from BEA's official machine-readable
    release-date JSON feed.

    This intentionally avoids pandas.read_html(), because the public
    schedule page may be rendered without a conventional HTML table.

    Only the two BEA releases used by the Investment OS event-quality
    check are retained:
    - Gross Domestic Product
    - Personal Income and Outlays
    """

    response = SESSION.get(
        "https://apps.bea.gov/"
        "API/signup/release_dates.json",
        timeout=(10, 60),
    )

    if not response.ok:
        raise RuntimeError(
            "BEA release-date JSON request failed: "
            f"status={response.status_code}; "
            f"body={response.text[:1000]}"
        )

    payload = response.json()

    if not isinstance(payload, dict):
        raise RuntimeError(
            "BEA release-date JSON returned "
            "an unexpected top-level structure"
        )

    required_releases = {
        "Gross Domestic Product",
        "Personal Income and Outlays",
    }

    events: list[dict[str, Any]] = []

    for release_name in required_releases:
        release_payload = payload.get(
            release_name
        )

        if not isinstance(
            release_payload,
            dict,
        ):
            LOGGER.warning(
                "BEA JSON does not contain %s",
                release_name,
            )
            continue

        release_dates = release_payload.get(
            "release_dates",
            [],
        )

        if not isinstance(
            release_dates,
            list,
        ):
            LOGGER.warning(
                "BEA JSON release_dates is not "
                "a list for %s",
                release_name,
            )
            continue

        for raw_timestamp in release_dates:
            timestamp = pd.to_datetime(
                raw_timestamp,
                errors="coerce",
                utc=True,
            )

            if pd.isna(timestamp):
                LOGGER.warning(
                    "Skipping invalid BEA date "
                    "%r for %s",
                    raw_timestamp,
                    release_name,
                )
                continue

            eastern_timestamp = (
                pd.Timestamp(timestamp)
                .tz_convert(
                    "America/New_York"
                )
            )

            events.append(
                {
                    "event_date":
                        eastern_timestamp
                        .date()
                        .isoformat(),
                    "asset":
                        "MARKET",
                    "event_type":
                        "macro_release",
                    "description":
                        release_name,
                    # Keep the existing source label so the
                    # production engine remains compatible.
                    "source":
                        "BEA_RELEASE_SCHEDULE",
                    "is_proxy":
                        False,
                }
            )

    output = (
        pd.DataFrame(
            events
        )
        .drop_duplicates(
            subset=[
                "event_date",
                "description",
                "source",
            ],
            keep="last",
        )
        .sort_values(
            [
                "event_date",
                "description",
            ]
        )
    )

    if output.empty:
        raise RuntimeError(
            "BEA machine-readable release feed "
            "returned no usable GDP or PCE events"
        )

    LOGGER.info(
        "Collected %s BEA GDP/PCE event rows; "
        "feed_last_updated=%s",
        len(output),
        payload.get(
            "file_last_updated",
            "unknown",
        ),
    )

    return output.to_dict(
        "records"
    )


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




TSMC_QUARTERLY_ROOT = (
    "https://investor.tsmc.com/english/"
    "quarterly-results"
)

MICRON_EVENTS_URL = (
    "https://investors.micron.com/"
    "events-and-presentations"
)

MONTH_PATTERN = (
    r"January|February|March|April|May|June|July|"
    r"August|September|October|November|December"
)


def normalize_document_text(
    text: str,
) -> str:
    normalized = (
        str(text)
        .replace("\u00a0", " ")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2019", "'")
        .replace("\u2212", "-")
    )

    return re.sub(
        r"[ \t]+",
        " ",
        normalized,
    )


def download_pdf_text(
    url: str,
) -> tuple[str, str]:
    response = SESSION.get(
        url,
        timeout=(10, 120),
    )

    if not response.ok:
        raise RuntimeError(
            "Official PDF request failed: "
            f"url={url}; "
            f"status={response.status_code}; "
            f"body={response.text[:500]}"
        )

    content = response.content

    if not content.startswith(b"%PDF"):
        raise RuntimeError(
            "Official document is not a PDF: "
            f"url={url}; "
            f"content_type={response.headers.get('Content-Type')}"
        )

    document_hash = hashlib.sha256(
        content
    ).hexdigest()

    reader = PdfReader(
        BytesIO(content)
    )

    pages = [
        page.extract_text() or ""
        for page in reader.pages
    ]

    text = normalize_document_text(
        "\n".join(pages)
    )

    if len(text.strip()) < 200:
        raise RuntimeError(
            "Official PDF has no usable text layer: "
            f"url={url}"
        )

    return text, document_hash


def parse_document_date(
    text: str,
) -> pd.Timestamp:
    match = re.search(
        rf"\b({MONTH_PATTERN})\s+"
        r"(\d{1,2}),\s+(20\d{2})\b",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        raise RuntimeError(
            "Unable to parse official document date"
        )

    parsed = pd.to_datetime(
        match.group(0),
        errors="coerce",
    )

    if pd.isna(parsed):
        raise RuntimeError(
            "Official document date is invalid: "
            f"{match.group(0)}"
        )

    return pd.Timestamp(parsed).normalize()


def tsmc_quarter_end(
    year: int,
    quarter: int,
) -> pd.Timestamp:
    quarter_month = {
        1: 3,
        2: 6,
        3: 9,
        4: 12,
    }[quarter]

    return (
        pd.Timestamp(
            year=year,
            month=quarter_month,
            day=1,
        )
        + pd.offsets.MonthEnd(0)
    )


def find_tsmc_presentation(
    year: int,
    quarter: int,
) -> Optional[dict[str, Any]]:
    page_url = (
        f"{TSMC_QUARTERLY_ROOT}/"
        f"{year}/q{quarter}"
    )

    response = SESSION.get(
        page_url,
        timeout=(10, 60),
    )

    if not response.ok:
        return None

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    for anchor in soup.find_all(
        "a",
        href=True,
    ):
        label = " ".join(
            anchor.get_text(
                " ",
                strip=True,
            ).split()
        ).lower()

        if "presentation material" not in label:
            continue

        return {
            "year": year,
            "quarter": quarter,
            "page_url": page_url,
            "pdf_url": urljoin(
                page_url,
                anchor["href"],
            ),
        }

    return None


def discover_latest_tsmc_quarter() -> dict[str, Any]:
    now_taipei = utc_now().tz_convert(
        "Asia/Taipei"
    )

    current_year = int(
        now_taipei.year
    )

    candidates: list[
        dict[str, Any]
    ] = []

    for year in range(
        current_year,
        current_year - 3,
        -1,
    ):
        for quarter in (
            4,
            3,
            2,
            1,
        ):
            result = find_tsmc_presentation(
                year,
                quarter,
            )

            if result is not None:
                candidates.append(
                    result
                )

    if not candidates:
        raise RuntimeError(
            "No TSMC quarterly presentation "
            "could be discovered"
        )

    latest = max(
        candidates,
        key=lambda item: (
            item["year"],
            item["quarter"],
        ),
    )

    previous = find_tsmc_presentation(
        int(latest["year"]) - 1,
        int(latest["quarter"]),
    )

    if previous is None:
        raise RuntimeError(
            "TSMC prior-year comparable "
            "presentation is unavailable"
        )

    return {
        "current": latest,
        "previous": previous,
    }


def parse_tsmc_net_revenue_usd_bn(
    text: str,
) -> float:
    patterns = (
        r"Net Revenue\s*"
        r"\(US\$\s*billions?\)\s*"
        r"([0-9]+(?:\.[0-9]+)?)",
        r"Net Revenue\s*"
        r"\(US\$ bn\)\s*"
        r"([0-9]+(?:\.[0-9]+)?)",
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        value = float(
            match.group(1)
        )

        if 1.0 < value < 200.0:
            return value

    raise RuntimeError(
        "Unable to parse TSMC net revenue "
        "from official presentation"
    )


def parse_tsmc_hpc_share_pct(
    text: str,
) -> float:
    lower_text = text.lower()
    marker = lower_text.find(
        "revenue by platform"
    )

    if marker < 0:
        raise RuntimeError(
            "TSMC Revenue by Platform "
            "section was not found"
        )

    segment = text[
        marker:
        marker + 3000
    ]

    patterns = (
        r"\bHPC\s*[\r\n ]*"
        r"([0-9]+(?:\.[0-9]+)?)\s*%",
        r"([0-9]+(?:\.[0-9]+)?)\s*%"
        r"[\r\n ]*\bHPC\b",
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            segment,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        value = float(
            match.group(1)
        )

        if 5.0 <= value <= 95.0:
            return value

    raise RuntimeError(
        "Unable to parse TSMC HPC share "
        "from Revenue by Platform chart"
    )


def collect_tsmc_hpc_growth_row(
    fetched_at: pd.Timestamp,
) -> dict[str, Any]:
    presentations = (
        discover_latest_tsmc_quarter()
    )

    current_meta = presentations[
        "current"
    ]

    previous_meta = presentations[
        "previous"
    ]

    current_text, current_hash = (
        download_pdf_text(
            current_meta[
                "pdf_url"
            ]
        )
    )

    previous_text, previous_hash = (
        download_pdf_text(
            previous_meta[
                "pdf_url"
            ]
        )
    )

    current_revenue = (
        parse_tsmc_net_revenue_usd_bn(
            current_text
        )
    )

    previous_revenue = (
        parse_tsmc_net_revenue_usd_bn(
            previous_text
        )
    )

    current_share = (
        parse_tsmc_hpc_share_pct(
            current_text
        )
    )

    previous_share = (
        parse_tsmc_hpc_share_pct(
            previous_text
        )
    )

    current_hpc_proxy = (
        current_revenue
        * current_share
        / 100.0
    )

    previous_hpc_proxy = (
        previous_revenue
        * previous_share
        / 100.0
    )

    if previous_hpc_proxy <= 0:
        raise RuntimeError(
            "TSMC prior-year HPC revenue "
            "proxy is not positive"
        )

    growth_pct = (
        100.0
        * (
            current_hpc_proxy
            / previous_hpc_proxy
            - 1.0
        )
    )

    if not (
        -80.0
        < growth_pct
        < 300.0
    ):
        raise RuntimeError(
            "TSMC HPC growth is outside "
            f"plausible bounds: {growth_pct}"
        )

    release_date = parse_document_date(
        current_text
    )

    release_timestamp = (
        release_date
        + pd.Timedelta(
            hours=16
        )
    ).tz_localize(
        "Asia/Taipei"
    ).tz_convert(
        "UTC"
    )

    return {
        "observation_date":
            tsmc_quarter_end(
                int(
                    current_meta[
                        "year"
                    ]
                ),
                int(
                    current_meta[
                        "quarter"
                    ]
                ),
            )
            .date()
            .isoformat(),
        "release_timestamp":
            release_timestamp.isoformat(),
        "effective_trade_date":
            resolve_effective_trade_date(
                release_timestamp
            ),
        "company":
            "TSMC",
        "metric":
            "tsmc_hpc_growth",
        "value":
            round(
                growth_pct,
                6,
            ),
        "score":
            round(
                logistic_score(
                    growth_pct,
                    midpoint=15.0,
                    scale=12.0,
                    higher_is_better=True,
                ),
                4,
            ),
        "source":
            "TSMC_QUARTERLY_PRESENTATION_"
            "HPC_REVENUE_PROXY",
        "is_proxy":
            True,
        "fetched_at":
            fetched_at.isoformat(),
        "source_id":
            (
                f"{current_hash},"
                f"{previous_hash}"
            ),
        "period_basis":
            "QUARTERLY_YOY",
        "metric_basis":
            "NET_REVENUE_X_HPC_SHARE_YOY",
        "document_url":
            current_meta[
                "pdf_url"
            ],
        "comparison_document_url":
            previous_meta[
                "pdf_url"
            ],
        "document_sha256":
            current_hash,
        "comparison_document_sha256":
            previous_hash,
        "current_revenue_usd_bn":
            round(
                current_revenue,
                6,
            ),
        "previous_revenue_usd_bn":
            round(
                previous_revenue,
                6,
            ),
        "current_hpc_share_pct":
            round(
                current_share,
                6,
            ),
        "previous_hpc_share_pct":
            round(
                previous_share,
                6,
            ),
        "current_hpc_revenue_proxy_usd_bn":
            round(
                current_hpc_proxy,
                6,
            ),
        "previous_hpc_revenue_proxy_usd_bn":
            round(
                previous_hpc_proxy,
                6,
            ),
    }


def parse_micron_event_timestamp(
    anchor: Any,
) -> Optional[pd.Timestamp]:
    pattern = re.compile(
        rf"\b({MONTH_PATTERN})\s+"
        r"(\d{1,2}),\s+(20\d{2})"
        r"(?:\s+at\s+"
        r"(\d{1,2}:\d{2})\s+"
        r"(AM|PM)\s+"
        r"(EDT|EST|MDT|MST|CDT|CST|PDT|PST))?",
        flags=re.IGNORECASE,
    )

    timezone_map = {
        "EDT": "America/New_York",
        "EST": "America/New_York",
        "MDT": "America/Denver",
        "MST": "America/Denver",
        "CDT": "America/Chicago",
        "CST": "America/Chicago",
        "PDT": "America/Los_Angeles",
        "PST": "America/Los_Angeles",
    }

    for text_node in anchor.find_all_previous(
        string=True,
        limit=120,
    ):
        candidate = " ".join(
            str(text_node).split()
        )

        match = pattern.search(
            candidate
        )

        if not match:
            continue

        date_text = (
            f"{match.group(1)} "
            f"{match.group(2)}, "
            f"{match.group(3)}"
        )

        if match.group(4):
            date_text += (
                f" {match.group(4)} "
                f"{match.group(5)}"
            )

        parsed = pd.to_datetime(
            date_text,
            errors="coerce",
        )

        if pd.isna(parsed):
            continue

        timestamp = pd.Timestamp(
            parsed
        )

        if match.group(6):
            timestamp = (
                timestamp
                .tz_localize(
                    timezone_map[
                        match.group(6).upper()
                    ]
                )
                .tz_convert(
                    "UTC"
                )
            )
        else:
            timestamp = (
                timestamp
                .normalize()
                .replace(
                    hour=16,
                    minute=30,
                )
                .tz_localize(
                    "America/New_York"
                )
                .tz_convert(
                    "UTC"
                )
            )

        return timestamp

    return None


def discover_latest_micron_prepared_remarks() -> dict[str, Any]:
    response = SESSION.get(
        MICRON_EVENTS_URL,
        timeout=(10, 60),
    )

    if not response.ok:
        raise RuntimeError(
            "Micron events page request failed: "
            f"status={response.status_code}; "
            f"body={response.text[:500]}"
        )

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    candidates: list[
        dict[str, Any]
    ] = []

    title_pattern = re.compile(
        r"\bQ([1-4])\s+(20\d{2})\s+"
        r"Prepared Remarks\b",
        flags=re.IGNORECASE,
    )

    for anchor in soup.find_all(
        "a",
        href=True,
    ):
        title = " ".join(
            anchor.get_text(
                " ",
                strip=True,
            ).split()
        )

        match = title_pattern.search(
            title
        )

        if not match:
            continue

        candidates.append(
            {
                "fiscal_quarter":
                    int(
                        match.group(1)
                    ),
                "fiscal_year":
                    int(
                        match.group(2)
                    ),
                "title":
                    title,
                "pdf_url":
                    urljoin(
                        MICRON_EVENTS_URL,
                        anchor[
                            "href"
                        ],
                    ),
                "release_timestamp":
                    parse_micron_event_timestamp(
                        anchor
                    ),
            }
        )

    if not candidates:
        raise RuntimeError(
            "No Micron Prepared Remarks "
            "links were discovered"
        )

    def candidate_key(
        item: dict[str, Any],
    ) -> tuple[int, int, int]:
        timestamp = item.get(
            "release_timestamp"
        )

        timestamp_value = (
            int(
                pd.Timestamp(
                    timestamp
                ).value
            )
            if timestamp is not None
            else 0
        )

        return (
            int(
                item[
                    "fiscal_year"
                ]
            ),
            int(
                item[
                    "fiscal_quarter"
                ]
            ),
            timestamp_value,
        )

    return max(
        candidates,
        key=candidate_key,
    )


def parse_micron_quantitative_growth(
    text: str,
) -> dict[str, Any]:
    normalized = normalize_document_text(
        text
    )

    patterns = (
        (
            "CDBU_REVENUE_YOY",
            r"(?:Core Data Center Business Unit "
            r"\(CDBU\)|CDBU).{0,500}?"
            r"(?:revenue\s+)?(?:was\s+)?"
            r"(?:up|increased|grew)\s+"
            r"([0-9]+(?:\.[0-9]+)?)%"
            r"\s+(?:year over year|year-over-year)",
            "YOY",
            25.0,
            20.0,
        ),
        (
            "CDBU_REVENUE_QOQ",
            r"(?:Core Data Center Business Unit "
            r"\(CDBU\)|CDBU).{0,500}?"
            r"(?:revenue\s+)?(?:was\s+)?"
            r"(?:up|increased|grew)\s+"
            r"([0-9]+(?:\.[0-9]+)?)%"
            r"\s+(?:sequentially|quarter over quarter|"
            r"quarter-over-quarter)",
            "QOQ",
            10.0,
            15.0,
        ),
        (
            "DATA_CENTER_REVENUE_YOY",
            r"data center revenue.{0,300}?"
            r"(?:was\s+)?(?:up|increased|grew)\s+"
            r"([0-9]+(?:\.[0-9]+)?)%"
            r"\s+(?:year over year|year-over-year)",
            "YOY",
            25.0,
            20.0,
        ),
        (
            "DATA_CENTER_REVENUE_QOQ",
            r"data center revenue.{0,300}?"
            r"(?:was\s+)?(?:up|increased|grew)\s+"
            r"([0-9]+(?:\.[0-9]+)?)%"
            r"\s+(?:sequentially|quarter over quarter|"
            r"quarter-over-quarter)",
            "QOQ",
            10.0,
            15.0,
        ),
        (
            "HBM_REVENUE_YOY",
            r"HBM(?:[0-9A-Za-z -]*)?\s+revenue"
            r".{0,300}?"
            r"(?:was\s+)?(?:up|increased|grew)\s+"
            r"([0-9]+(?:\.[0-9]+)?)%"
            r"\s+(?:year over year|year-over-year)",
            "YOY",
            25.0,
            20.0,
        ),
        (
            "HBM_REVENUE_QOQ",
            r"HBM(?:[0-9A-Za-z -]*)?\s+revenue"
            r".{0,300}?"
            r"(?:was\s+)?(?:up|increased|grew)\s+"
            r"([0-9]+(?:\.[0-9]+)?)%"
            r"\s+(?:sequentially|quarter over quarter|"
            r"quarter-over-quarter)",
            "QOQ",
            10.0,
            15.0,
        ),
    )

    for (
        basis,
        pattern,
        period_basis,
        midpoint,
        scale,
    ) in patterns:
        match = re.search(
            pattern,
            normalized,
            flags=(
                re.IGNORECASE
                | re.DOTALL
            ),
        )

        if not match:
            continue

        growth_pct = float(
            match.group(1)
        )

        if not (
            -100.0
            < growth_pct
            < 1000.0
        ):
            continue

        return {
            "value":
                growth_pct,
            "metric_basis":
                basis,
            "period_basis":
                period_basis,
            "score":
                logistic_score(
                    growth_pct,
                    midpoint=midpoint,
                    scale=scale,
                    higher_is_better=True,
                ),
            "matched_text":
                " ".join(
                    match.group(0).split()
                )[:1000],
        }

    raise RuntimeError(
        "Micron Prepared Remarks did not "
        "contain a supported quantified "
        "Data Center or HBM growth statement"
    )


def collect_micron_dc_hbm_row(
    fetched_at: pd.Timestamp,
) -> dict[str, Any]:
    metadata = (
        discover_latest_micron_prepared_remarks()
    )

    text, document_hash = (
        download_pdf_text(
            metadata[
                "pdf_url"
            ]
        )
    )

    parsed = (
        parse_micron_quantitative_growth(
            text
        )
    )

    release_timestamp = metadata.get(
        "release_timestamp"
    )

    if release_timestamp is None:
        release_date = (
            parse_document_date(
                text
            )
        )

        release_timestamp = (
            release_date
            + pd.Timedelta(
                hours=16,
                minutes=30,
            )
        ).tz_localize(
            "America/New_York"
        ).tz_convert(
            "UTC"
        )

    release_timestamp = pd.Timestamp(
        release_timestamp
    )

    if release_timestamp.tzinfo is None:
        release_timestamp = (
            release_timestamp
            .tz_localize(
                "UTC"
            )
        )
    else:
        release_timestamp = (
            release_timestamp
            .tz_convert(
                "UTC"
            )
        )

    return {
        "observation_date":
            release_timestamp
            .date()
            .isoformat(),
        "release_timestamp":
            release_timestamp.isoformat(),
        "effective_trade_date":
            resolve_effective_trade_date(
                release_timestamp
            ),
        "company":
            "MU",
        "metric":
            "micron_dc_hbm_score",
        "value":
            round(
                float(
                    parsed[
                        "value"
                    ]
                ),
                6,
            ),
        "score":
            round(
                float(
                    parsed[
                        "score"
                    ]
                ),
                4,
            ),
        "source":
            "MICRON_PREPARED_REMARKS_"
            "QUANTITATIVE_PROXY",
        "is_proxy":
            True,
        "fetched_at":
            fetched_at.isoformat(),
        "source_id":
            document_hash,
        "period_basis":
            parsed[
                "period_basis"
            ],
        "metric_basis":
            parsed[
                "metric_basis"
            ],
        "document_url":
            metadata[
                "pdf_url"
            ],
        "document_sha256":
            document_hash,
        "document_title":
            metadata[
                "title"
            ],
        "fiscal_year":
            metadata[
                "fiscal_year"
            ],
        "fiscal_quarter":
            metadata[
                "fiscal_quarter"
            ],
        "matched_text":
            parsed[
                "matched_text"
            ],
        "observation_date_is_release_proxy":
            True,
    }



def collect_ai_cycle() -> None:
    """
    Collect four AI-cycle inputs:

    1. NVIDIA quarterly total-revenue YoY proxy.
    2. Hyperscaler CapEx YoY, amount weighted.
    3. TSMC HPC revenue-growth proxy from official presentations.
    4. Micron Data Center / HBM growth from official Prepared Remarks.

    TSMC and Micron remain proxy rows because their values are derived
    from official disclosure text rather than standardized XBRL fields.
    """

    fetched_at = utc_now()
    rows: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # NVIDIA quarterly revenue YoY proxy
    # --------------------------------------------------------
    nvda_facts = fetch_company_facts(
        COMPANIES["NVDA"]
    )

    nvda_pair = comparable_duration_pair(
        nvda_facts,
        REVENUE_CONCEPTS,
        min_duration_days=70,
        max_duration_days=120,
    )

    if nvda_pair is not None:
        release = (
            conservative_release_timestamp(
                nvda_pair["filed"]
            )
        )

        value = float(
            nvda_pair["yoy"]
        )

        rows.append(
            {
                "observation_date":
                    nvda_pair[
                        "period_end"
                    ]
                    .date()
                    .isoformat(),
                "release_timestamp":
                    release.isoformat(),
                "effective_trade_date":
                    resolve_effective_trade_date(
                        release
                    ),
                "company":
                    "NVDA",
                "metric":
                    "nvidia_revenue_yoy_proxy",
                "value":
                    round(
                        value,
                        6,
                    ),
                # Wider scale avoids near-automatic 99 scores.
                "score":
                    round(
                        logistic_score(
                            value,
                            midpoint=25.0,
                            scale=25.0,
                            higher_is_better=True,
                        ),
                        4,
                    ),
                "source":
                    "SEC_XBRL_QUARTERLY_"
                    "TOTAL_REVENUE_PROXY",
                "is_proxy":
                    True,
                "fetched_at":
                    fetched_at.isoformat(),
                "source_id":
                    nvda_pair[
                        "accession"
                    ],
                "period_basis":
                    "QUARTERLY",
            }
        )

    # --------------------------------------------------------
    # Hyperscaler CapEx: same duration basis + dollar weighting
    # --------------------------------------------------------
    selected_basis: Optional[str] = None
    selected_results: list[
        tuple[str, dict[str, Any]]
    ] = []

    duration_buckets = (
        (
            "QUARTERLY",
            70,
            120,
        ),
        (
            "YTD",
            150,
            300,
        ),
    )

    for basis, minimum, maximum in duration_buckets:
        candidates: list[
            tuple[str, dict[str, Any]]
        ] = []

        for ticker in (
            "MSFT",
            "META",
            "GOOGL",
            "AMZN",
        ):
            try:
                facts = fetch_company_facts(
                    COMPANIES[
                        ticker
                    ]
                )

                pair = comparable_duration_pair(
                    facts,
                    CAPEX_CONCEPTS,
                    min_duration_days=minimum,
                    max_duration_days=maximum,
                )

                if pair is None:
                    continue

                candidates.append(
                    (
                        ticker,
                        pair,
                    )
                )

            except Exception:
                LOGGER.exception(
                    "Unable to collect %s "
                    "CapEx comparison for %s",
                    basis,
                    ticker,
                )

        if len(candidates) >= 3:
            selected_basis = basis
            selected_results = candidates
            break

    if selected_results:
        aggregate_current = sum(
            abs(
                float(
                    result[
                        "current_value"
                    ]
                )
            )
            for _, result
            in selected_results
        )

        aggregate_previous = sum(
            abs(
                float(
                    result[
                        "previous_value"
                    ]
                )
            )
            for _, result
            in selected_results
        )

        if aggregate_previous > 0:
            value = (
                100.0
                * (
                    aggregate_current
                    / aggregate_previous
                    - 1.0
                )
            )

            filed = max(
                pd.Timestamp(
                    result[
                        "filed"
                    ]
                )
                for _, result
                in selected_results
            )

            release = (
                conservative_release_timestamp(
                    filed
                )
            )

            rows.append(
                {
                    "observation_date":
                        max(
                            pd.Timestamp(
                                result[
                                    "period_end"
                                ]
                            )
                            for _, result
                            in selected_results
                        )
                        .date()
                        .isoformat(),
                    "release_timestamp":
                        release.isoformat(),
                    "effective_trade_date":
                        resolve_effective_trade_date(
                            release
                        ),
                    "company":
                        "_".join(
                            ticker
                            for ticker, _
                            in selected_results
                        ),
                    "metric":
                        "hyperscaler_capex_yoy",
                    "value":
                        round(
                            value,
                            6,
                        ),
                    "score":
                        round(
                            logistic_score(
                                value,
                                midpoint=20.0,
                                scale=25.0,
                                higher_is_better=True,
                            ),
                            4,
                        ),
                    "source":
                        "SEC_XBRL_"
                        f"{selected_basis}_"
                        "CAPEX_AMOUNT_WEIGHTED_PROXY",
                    "is_proxy":
                        True,
                    "fetched_at":
                        fetched_at.isoformat(),
                    "source_id":
                        ",".join(
                            sorted(
                                {
                                    str(
                                        result.get(
                                            "accession",
                                            "",
                                        )
                                    )
                                    for _, result
                                    in selected_results
                                }
                                - {""}
                            )
                        ),
                    "period_basis":
                        selected_basis,
                    "constituent_count":
                        len(
                            selected_results
                        ),
                    "aggregate_current_usd":
                        round(
                            aggregate_current,
                            2,
                        ),
                    "aggregate_previous_usd":
                        round(
                            aggregate_previous,
                            2,
                        ),
                }
            )

    # --------------------------------------------------------
    # TSMC HPC and Micron DC/HBM official-disclosure proxies
    # --------------------------------------------------------
    for source_name, collector in (
        (
            "TSMC HPC",
            collect_tsmc_hpc_growth_row,
        ),
        (
            "Micron DC/HBM",
            collect_micron_dc_hbm_row,
        ),
    ):
        try:
            row = collector(
                fetched_at
            )

            rows.append(
                row
            )

            LOGGER.info(
                "%s AI Cycle row accepted: "
                "value=%s score=%s basis=%s",
                source_name,
                row.get(
                    "value"
                ),
                row.get(
                    "score"
                ),
                row.get(
                    "metric_basis"
                ),
            )

        except Exception:
            LOGGER.exception(
                "%s AI Cycle source failed",
                source_name,
            )

    if not rows:
        raise RuntimeError(
            "No usable AI Cycle proxy rows "
            "were collected"
        )

    append_pit_csv(
        AI_CYCLE_FILE,
        pd.DataFrame(
            rows
        ),
        (
            "observation_date",
            "company",
            "metric",
            "source",
        ),
        (
            "effective_trade_date",
            "company",
            "metric",
        ),
    )


def normalize_yfinance_iv_percent(
    raw_value: float,
) -> float:
    """
    Convert a yfinance implied-volatility value into percentage points.

    Normal yfinance format:
        0.1963 -> 19.63

    Some yfinance/parser versions may expose an additional 1/100
    scaling:
        0.001963 -> 19.63

    Because this collector is specifically for an equity ETF options
    proxy, a final result below 3 percentage points is treated as a
    likely unit-scaling error and multiplied by 100 once more.
    """

    value = float(
        raw_value
    )

    if (
        not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            "Implied volatility must be "
            f"positive and finite: {raw_value}"
        )

    percent_value = (
        value
        * 100.0
    )

    if (
        0
        < percent_value
        < 3.0
    ):
        percent_value *= 100.0

    if not (
        3.0
        <= percent_value
        < 500.0
    ):
        raise ValueError(
            "Normalized implied volatility "
            "is outside plausible bounds: "
            f"raw={raw_value}, "
            f"percent={percent_value}"
        )

    return float(
        percent_value
    )


def repair_legacy_iv30_units() -> None:
    """
    Repair historical IV30 values saved with decimal or double-decimal
    scaling.

    Examples:
        0.1963   -> 19.63
        0.001963 -> 19.63
    """

    if not POSITIONING_FILE.exists():
        return

    frame = pd.read_csv(
        POSITIONING_FILE
    )

    required = {
        "metric",
        "value",
        "source",
    }

    if not required.issubset(
        frame.columns
    ):
        return

    metric = (
        frame["metric"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    source = (
        frame["source"]
        .fillna("")
        .astype(str)
    )

    values = pd.to_numeric(
        frame["value"],
        errors="coerce",
    )

    target_mask = (
        metric.eq("iv30")
        & source.str.startswith(
            "YFINANCE_"
        )
        & values.gt(0)
        & values.lt(3)
    )

    if not target_mask.any():
        return

    repaired_count = 0

    for row_index in frame.index[
        target_mask
    ]:
        raw_value = pd.to_numeric(
            frame.at[
                row_index,
                "value",
            ],
            errors="coerce",
        )

        if pd.isna(raw_value):
            continue

        frame.at[
            row_index,
            "value",
        ] = normalize_yfinance_iv_percent(
            float(raw_value)
        )

        repaired_count += 1

    if repaired_count == 0:
        return

    frame.to_csv(
        POSITIONING_FILE,
        index=False,
    )

    LOGGER.info(
        "Repaired %s legacy IV30 rows",
        repaired_count,
    )


def collect_options_metrics() -> None:
    """
    Collect a free QQQ options-volatility proxy.

    - IV30: mean ATM call/put implied volatility at the listed expiry
      closest to 30 calendar days.
    - RV20: annualized standard deviation of the latest 20 daily
      log returns.

    yfinance does not provide a verified delta, so this collector
    must not enable an exact strike recommendation.
    """

    repair_legacy_iv30_units()

    ticker = yf.Ticker(
        OPTIONS_UNDERLYING
    )

    fetched_at = utc_now()

    history = ticker.history(
        period="6mo",
        interval="1d",
        auto_adjust=False,
    )

    if (
        history.empty
        or "Close" not in history.columns
    ):
        raise RuntimeError(
            f"No price history for "
            f"{OPTIONS_UNDERLYING}"
        )

    close = pd.to_numeric(
        history["Close"],
        errors="coerce",
    ).dropna()

    if len(close) < 21:
        raise RuntimeError(
            "Fewer than 21 closes available "
            "for RV20"
        )

    spot = float(
        close.iloc[-1]
    )

    price_date = pd.Timestamp(
        close.index[-1]
    )

    if price_date.tzinfo is not None:
        price_date = (
            price_date
            .tz_localize(None)
        )

    log_returns = np.log(
        close
        / close.shift(1)
    ).dropna()

    rv20 = float(
        log_returns
        .tail(20)
        .std(
            ddof=1
        )
        * math.sqrt(
            252.0
        )
        * 100.0
    )

    expirations = [
        pd.Timestamp(
            value
        )
        for value in ticker.options
    ]

    if not expirations:
        raise RuntimeError(
            f"No listed options for "
            f"{OPTIONS_UNDERLYING}"
        )

    today_et = (
        fetched_at
        .tz_convert(
            "America/New_York"
        )
        .tz_localize(None)
        .normalize()
    )

    candidates = [
        expiry
        for expiry in expirations
        if 14
        <= (
            expiry.normalize()
            - today_et
        ).days
        <= 60
    ]

    if not candidates:
        candidates = [
            expiry
            for expiry in expirations
            if expiry.normalize()
            > today_et
        ]

    if not candidates:
        raise RuntimeError(
            "No future option expiration found"
        )

    expiration = min(
        candidates,
        key=lambda expiry: abs(
            (
                expiry.normalize()
                - today_et
            ).days
            - OPTIONS_TARGET_DTE
        ),
    )

    days_to_expiry = int(
        (
            expiration.normalize()
            - today_et
        ).days
    )

    chain = ticker.option_chain(
        expiration.date().isoformat()
    )

    iv_values: list[float] = []
    atm_strikes: list[float] = []

    for side in (
        chain.calls,
        chain.puts,
    ):
        if side is None or side.empty:
            continue

        usable = side.copy()

        usable["strike"] = pd.to_numeric(
            usable["strike"],
            errors="coerce",
        )

        usable[
            "impliedVolatility"
        ] = pd.to_numeric(
            usable[
                "impliedVolatility"
            ],
            errors="coerce",
        )

        usable = usable.dropna(
            subset=[
                "strike",
                "impliedVolatility",
            ]
        )

        usable = usable[
            usable[
                "impliedVolatility"
            ].between(
                0.001,
                5.0,
            )
        ]

        if usable.empty:
            continue

        row = usable.loc[
            (
                usable[
                    "strike"
                ]
                - spot
            )
            .abs()
            .idxmin()
        ]

        iv_values.append(
            normalize_yfinance_iv_percent(
                float(
                    row[
                        "impliedVolatility"
                    ]
                )
            )
        )

        atm_strikes.append(
            float(
                row[
                    "strike"
                ]
            )
        )

    if not iv_values:
        raise RuntimeError(
            "No usable ATM implied volatility"
        )

    iv30 = float(
        np.mean(
            iv_values
        )
    )

    # Final guard immediately before persistence.
    # The saved IV30 must always be percentage points,
    # such as 19.63 rather than 0.1963.
    if iv30 < 3.0:
        iv30 = normalize_yfinance_iv_percent(
            iv30
        )

    if not (
        3.0 <= iv30 < 500
        and 0 < rv20 < 500
    ):
        raise RuntimeError(
            "Options volatility metrics are "
            f"outside plausible bounds: "
            f"IV={iv30}, RV={rv20}"
        )

    effective_trade_date = (
        resolve_effective_trade_date(
            fetched_at
        )
    )

    common = {
        "observation_date":
            price_date
            .date()
            .isoformat(),
        "release_timestamp":
            fetched_at.isoformat(),
        "effective_trade_date":
            effective_trade_date,
        "source":
            "YFINANCE_QQQ_OPTIONS_PROXY",
        "is_proxy":
            True,
        "fetched_at":
            fetched_at.isoformat(),
        "underlying":
            OPTIONS_UNDERLYING,
        "option_expiration":
            expiration
            .date()
            .isoformat(),
        "days_to_expiry":
            days_to_expiry,
        "atm_strike":
            round(
                float(
                    np.mean(
                        atm_strikes
                    )
                ),
                4,
            ),
    }

    rows = pd.DataFrame(
        [
            {
                **common,
                "metric":
                    "iv30",
                "value":
                    round(
                        iv30,
                        6,
                    ),
                "score":
                    np.nan,
            },
            {
                **common,
                "metric":
                    "rv20",
                "value":
                    round(
                        rv20,
                        6,
                    ),
                "score":
                    np.nan,
            },
        ]
    )

    append_pit_csv(
        POSITIONING_FILE,
        rows,
        (
            "effective_trade_date",
            "metric",
            "source",
            "underlying",
        ),
    )



def main() -> None:
    validate_environment()

    results = {
        "SEC TTM Valuation":
            run_safely(
                "SEC TTM Valuation",
                collect_sec_valuation,
            ),
        "GDPNow Vintage":
            run_safely(
                "GDPNow Vintage",
                collect_gdpnow,
            ),
        "Cboe Put/Call":
            run_safely(
                "Cboe Put/Call",
                collect_cboe_put_call,
            ),
        "CFTC":
            run_safely(
                "CFTC",
                collect_cftc_positioning,
            ),
        "Options IV/RV":
            run_safely(
                "Options IV/RV",
                collect_options_metrics,
            ),
        "Events":
            run_safely(
                "Events",
                collect_events,
            ),
        "AI Cycle":
            run_safely(
                "AI Cycle",
                collect_ai_cycle,
            ),
    }

    LOGGER.info(
        "Collector summary: %s",
        results,
    )

    if (
        not VALUATION_FILE.exists()
        or VALUATION_FILE.stat().st_size
        == 0
    ):
        raise RuntimeError(
            "Required valuation_pit.csv "
            "was not generated"
        )

if __name__ == "__main__":
    main()
