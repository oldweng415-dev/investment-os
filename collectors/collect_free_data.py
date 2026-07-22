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
                    "lev_money_positions_long_all",
                    "lev_money_positions_short_all",
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

    response.raise_for_status()

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
        "lev_money_positions_long_all",
        "lev_money_positions_short_all",
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
        "lev_money_positions_long_all",
        "lev_money_positions_short_all",
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
                "lev_money_positions_long_all"
            ]
            - grouped[
                "lev_money_positions_short_all"
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
    Parse the official BEA full-year release schedule.

    Both the current and next year are requested explicitly, which
    avoids relying on a page table whose date column omits the year.
    """

    events: list[dict[str, Any]] = []
    current_year = pd.Timestamp.today().year

    month_pattern = (
        r"January|February|March|April|May|June|"
        r"July|August|September|October|November|December"
    )

    for year in (
        current_year,
        current_year + 1,
    ):
        response = SESSION.get(
            "https://www.bea.gov/"
            f"news/schedule/full/{year}",
            timeout=(10, 60),
        )

        response.raise_for_status()

        tables = pd.read_html(
            StringIO(
                response.text
            )
        )

        for table in tables:
            if table.empty:
                continue

            if isinstance(
                table.columns,
                pd.MultiIndex,
            ):
                table.columns = [
                    " ".join(
                        str(part).strip()
                        for part in column
                        if str(part).strip()
                        and str(part).lower()
                        != "nan"
                    )
                    for column in table.columns
                ]
            else:
                table.columns = [
                    str(column).strip()
                    for column in table.columns
                ]

            for _, row in table.iterrows():
                cells = [
                    str(value).strip()
                    for value in row.tolist()
                    if pd.notna(value)
                    and str(value).strip()
                    and str(value).strip().lower()
                    != "nan"
                ]

                if not cells:
                    continue

                row_text = " | ".join(
                    cells
                )

                date_match = re.search(
                    rf"\b({month_pattern})\s+"
                    r"(\d{1,2})\b",
                    row_text,
                    flags=re.IGNORECASE,
                )

                if not date_match:
                    continue

                event_date = pd.to_datetime(
                    f"{date_match.group(1)} "
                    f"{date_match.group(2)} "
                    f"{year}",
                    errors="coerce",
                )

                if pd.isna(
                    event_date
                ):
                    continue

                candidates = [
                    cell
                    for cell in cells
                    if not re.search(
                        rf"\b({month_pattern})\s+"
                        r"\d{1,2}\b",
                        cell,
                        flags=re.IGNORECASE,
                    )
                    and cell.lower()
                    not in {
                        "news",
                        "data",
                    }
                    and not re.fullmatch(
                        r"\d{1,2}:\d{2}\s*"
                        r"(?:am|pm)?",
                        cell,
                        flags=re.IGNORECASE,
                    )
                ]

                description = (
                    max(
                        candidates,
                        key=len,
                    )
                    if candidates
                    else "BEA economic release"
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
                            "macro_release",
                        "description":
                            description,
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
            "event_date"
        )
    )

    if output.empty:
        raise RuntimeError(
            "BEA schedule parser returned "
            "no usable events"
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



def collect_ai_cycle() -> None:
    """
    Collect two free AI-cycle proxies with conservative calibration:

    1. NVIDIA quarterly total-revenue YoY proxy.
    2. Hyperscaler CapEx YoY using a common duration basis and
       aggregate-dollar weighting rather than a simple average.

    TSMC HPC and Micron HBM remain unavailable until reliable
    machine-readable disclosures are supplied.
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
            float(
                row[
                    "impliedVolatility"
                ]
            )
            * 100.0
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

    if not (
        0 < iv30 < 500
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
