import os
import json
import logging
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from scipy.stats import mstats

# ==========================================
# 1. Configuration & Data Classes
# ==========================================
@dataclass
class Config:
    data_mode: str = "live_public"
    start_date: str = "2012-01-01"
    rolling_percentile_years: int = 10
    min_percentile_years: int = 3
    annual_borrow_rate: float = 0.04
    max_target_margin_pct: float = 20.0
    data_directory: str = "data"
    output_directory: str = "output"
    
    max_staleness_days: Dict[str, int] = field(default_factory=lambda: {
        "market_price": 5,
        "daily_market_indicator": 7,
        "weekly_macro": 14,
        "monthly_macro": 45,
        "quarterly_fundamental": 120
    })

    module_base_weights: Dict[str, float] = field(default_factory=lambda: {
        "market_regime": 0.21, "ai_cycle": 0.14, "valuation": 0.12,
        "macro": 0.17, "liquidity": 0.19, "positioning": 0.17
    })

cfg = Config()

# ==========================================
# 2. Shared Math & Statistics Functions
# ==========================================
def winsorize_series(series: pd.Series, limits: tuple = (0.01, 0.01)) -> pd.Series:
    """將資料限制於 1%~99% 極端值以內"""
    clean_series = series.dropna()
    if len(clean_series) < 10:
        return series
    winsorized = mstats.winsorize(clean_series, limits=limits)
    res = pd.Series(winsorized, index=clean_series.index)
    return series.copy().update(res)

def ewma_smooth(series: pd.Series, span: int = 5) -> pd.Series:
    """高頻資料的 5 日平滑處理"""
    return series.ewm(span=span, adjust=False).mean()

def rolling_percentile(series: pd.Series, window_years: int = cfg.rolling_percentile_years) -> pd.Series:
    """使用滾動視窗計算百分位數，確保無未來函數 (point-in-time)"""
    window_days = window_years * 252
    def calc_pct(x):
        if len(x.dropna()) < (cfg.min_percentile_years * 252):
            return np.nan
        return (x < x.iloc[-1]).mean() * 100
    return series.rolling(window=window_days, min_periods=cfg.min_percentile_years * 252).apply(calc_pct, raw=False)

def inverse_percentile_score(percentile_series: pd.Series) -> pd.Series:
    """將越高越差的指標 (如 OAS, Claims) 轉換為 0-100 分數"""
    return 100.0 - percentile_series

def safe_weighted_average(scores: dict, weights: dict) -> Tuple[float, dict, float]:
    """動態權重重配 (Dynamic Reweight) 且計算 coverage ratio"""
    valid_scores = {k: v for k, v in scores.items() if v is not None and not np.isnan(v)}
    if not valid_scores:
        return 0.0, {}, 0.0
    
    total_original_weight = sum(weights[k] for k in valid_scores.keys())
    coverage_ratio = total_original_weight / sum(weights.values())
    
    effective_weights = {k: weights[k] / total_original_weight for k in valid_scores.keys()}
    final_score = sum(valid_scores[k] * effective_weights[k] for k in valid_scores.keys())
    
    return np.clip(final_score, 0, 100), effective_weights, coverage_ratio

# ==========================================
# 3. Data Loading & Alignment
# ==========================================
def load_yfinance_data() -> pd.DataFrame:
    tickers = ["SPY", "QQQ", "SOXX", "IWM", "MDY", "SMH", "HYG", "NVDA", "^VIX", "^VIX3M", "^SKEW", "^VVIX", "^TNX"]
    df = yf.download(tickers, start=cfg.start_date, progress=False)['Close']
    return df

def load_fred_data() -> pd.DataFrame:
    series = {
        'BAMLH0A0HYM2': 'hy_oas',
        'WALCL': 'walcl',              # Mil USD
        'RRPONTSYD': 'rrp',            # Bil USD
        'WTREGEN': 'tga',              # Bil USD
        'WRESBAL': 'reserves',         # Bil USD
        'M2REAL': 'real_m2',           # Bil USD
        'ICSA': 'claims',
        'UNRATE': 'unrate',
        'PCEPILFE': 'core_pce',
        'DFII10': 'real_10y'
    }
    df = web.DataReader(list(series.keys()), 'fred', cfg.start_date)
    df.rename(columns=series, inplace=True)
    
    # Unit standardization (convert all to Billion USD)
    if 'walcl' in df.columns:
        df['walcl'] = df['walcl'] / 1000.0
    return df

def align_data_asof(market_df: pd.DataFrame, fred_df: pd.DataFrame) -> pd.DataFrame:
    """使用 merge_asof 對齊資料，避免無限制 ffill"""
    market_df = market_df.reset_index().rename(columns={'Date': 'date'})
    fred_df = fred_df.reset_index().rename(columns={'DATE': 'date'})
    
    merged = pd.merge_asof(market_df.sort_values('date'), 
                           fred_df.sort_values('date'), 
                           on='date', 
                           direction='backward')
    return merged.set_index('date')

# ==========================================
# 4. Module Calculations
# ==========================================
def calculate_market_regime(df: pd.DataFrame, today: pd.Series) -> dict:
    """Market Regime: SPY/QQQ 趨勢, 廣度 Proxy, 信用 OAS"""
    # 趨勢 (tanh 轉換)
    spy_200 = df['SPY'].rolling(200).mean().iloc[-1]
    qqq_200 = df['QQQ'].rolling(200).mean().iloc[-1]
    
    def get_trend_score(price, ma):
        if pd.isna(price) or pd.isna(ma): return 50
        z = np.log(price / ma) / (df['SPY'].pct_change().std() * np.sqrt(63)) # approx volatility
        return 50 + 25 * np.tanh(z)
    
    spy_trend = get_trend_score(today['SPY'], spy_200)
    qqq_trend = get_trend_score(today['QQQ'], qqq_200)
    
    # Breadth (Proxy)
    etfs = ['SPY', 'QQQ', 'SOXX', 'IWM', 'MDY', 'SMH']
    above_200ma_count = sum(1 for etf in etfs if today.get(etf, 0) > df[etf].rolling(200).mean().iloc[-1])
    breadth_score = (above_200ma_count / len(etfs)) * 100
    
    # Credit (OAS)
    oas_pct = rolling_percentile(df['hy_oas']).iloc[-1]
    credit_score = 100 - oas_pct if not pd.isna(oas_pct) else 50
    
    mr_score = 0.30*spy_trend + 0.30*qqq_trend + 0.20*breadth_score + 0.20*credit_score
    return {"score": np.clip(mr_score, 0, 100), "is_proxy": True, "breadth_score": breadth_score, "hy_oas_pct": oas_pct}

def calculate_liquidity(df: pd.DataFrame, today: pd.Series) -> dict:
    """Liquidity: Net Fed Liquidity"""
    df['nfl'] = df['walcl'] - df['tga'] - df['rrp']
    nfl_13w_change = df['nfl'].diff(65)
    
    nfl_pct = rolling_percentile(nfl_13w_change).iloc[-1]
    rrp_inverse = 100 - rolling_percentile(df['rrp'].diff(65)).iloc[-1]
    tga_inverse = 100 - rolling_percentile(df['tga'].diff(65)).iloc[-1]
    
    nfl_pct = nfl_pct if not pd.isna(nfl_pct) else 50
    rrp_inverse = rrp_inverse if not pd.isna(rrp_inverse) else 50
    tga_inverse = tga_inverse if not pd.isna(tga_inverse) else 50
    
    liq_score = 0.50*nfl_pct + 0.25*rrp_inverse + 0.25*tga_inverse
    return {"score": np.clip(liq_score, 0, 100), "nfl_13w_pct": nfl_pct, "nfl_val": today.get('nfl')}

def calculate_macro(df: pd.DataFrame, fred_weekly: pd.DataFrame, today: pd.Series) -> dict:
    """Macro: Claims (Calculated on weekly), Sahm Gap"""
    # Sahm Gap
    unrate_3m = df['unrate'].rolling(90).mean()
    unrate_12m_min = df['unrate'].rolling(365).min()
    sahm_gap = unrate_3m - unrate_12m_min
    sahm_score = 100 - rolling_percentile(sahm_gap).iloc[-1]
    
    # Claims (Strictly on weekly data)
    claims_4w = fred_weekly['claims'].rolling(4).mean()
    claims_26w = fred_weekly['claims'].rolling(26).mean()
    claims_ratio = claims_4w / claims_26w
    claims_score = 100 - rolling_percentile(claims_ratio).iloc[-1]
    
    sahm_score = sahm_score if not pd.isna(sahm_score) else 50
    claims_score = claims_score if not pd.isna(claims_score) else 50
    
    mac_score = 0.5 * sahm_score + 0.5 * claims_score
    return {"score": np.clip(mac_score, 0, 100)}

def calculate_positioning(df: pd.DataFrame, today: pd.Series) -> dict:
    """Positioning: VIX Term Structure, SKEW, VVIX"""
    vix = today.get('^VIX', 20)
    vix3m = today.get('^VIX3M', 20)
    
    # Term structure: Contango = Good (100), Backwardation = Bad (0)
    vix_ratio = vix / vix3m if vix3m > 0 else 1.0
    term_score = 100 - np.clip((vix_ratio - 0.9) * 500, 0, 100) 
    
    skew_pct = 100 - rolling_percentile(df['^SKEW']).iloc[-1]
    vvix_pct = 100 - rolling_percentile(df['^VVIX']).iloc[-1]
    
    term_score = term_score if not pd.isna(term_score) else 50
    skew_pct = skew_pct if not pd.isna(skew_pct) else 50
    vvix_pct = vvix_pct if not pd.isna(vvix_pct) else 50
    
    pos_score = 0.4*term_score + 0.3*skew_pct + 0.3*vvix_pct
    
    is_backwardation = vix > vix3m
    return {"score": np.clip(pos_score, 0, 100), "vix": vix, "is_backwardation": is_backwardation}

# ==========================================
# 5. Core Engine & Main Execution
# ==========================================
def main():
    print("啟動 Investment OS Live Engine...")
    os.makedirs(cfg.data_directory, exist_ok=True)
    os.makedirs(cfg.output_directory, exist_ok=True)
    
    # 1. Load Data
    df_mkt = load_yfinance_data()
    df_fred = load_fred_data()
    df = align_data_asof(df_mkt, df_fred)
    today = df.iloc[-1]
    
    # 2. Calculate Modules
    mr_data = calculate_market_regime(df, today)
    liq_data = calculate_liquidity(df, today)
    mac_data = calculate_macro(df, df_fred, today) # Pass raw fred for weekly claims
    pos_data = calculate_positioning(df, today)
    
    # PIT CSV Data (Mock check for demonstration)
    val_score = None # unavailable by default if no CSV
    ai_score = None  # unavailable by default if no CSV
    
    modules = {
        "market_regime": mr_data['score'],
        "ai_cycle": ai_score,
        "valuation": val_score,
        "macro": mac_data['score'],
        "liquidity": liq_data['score'],
        "positioning": pos_data['score']
    }
    
    # 3. Dynamic Reweighting
    buy_base, _, cov_ratio = safe_weighted_average(modules, {
        "market_regime": 0.25, "ai_cycle": 0.15, "valuation": 0.20,
        "macro": 0.10, "liquidity": 0.10, "positioning": 0.20
    })
    
    risk_support, _, _ = safe_weighted_average(modules, {
        "market_regime": 0.30, "ai_cycle": 0.10, "valuation": 0.05,
        "macro": 0.20, "liquidity": 0.25, "positioning": 0.10
    })
    
    margin_base, _, _ = safe_weighted_average(modules, {
        "market_regime": 0.25, "ai_cycle": 0.05, "valuation": 0.20,
        "macro": 0.10, "liquidity": 0.25, "positioning": 0.15
    })
    
    # 4. Modifiers & Penalties
    b_panic = 10 if (pos_data['score'] >= 80 and risk_support > 25) else 0
    p_euphoria = 10 if (pos_data['score'] <= 20 and (val_score is not None and val_score <= 40)) else 0
    buy_score = np.clip(buy_base + b_panic - p_euphoria, 0, 100)
    
    risk_score = 100 - risk_support
    risk_overrides = []
    
    if mr_data['hy_oas_pct'] > 90:
        risk_score += 10
        risk_overrides.append("HY_OAS_Extreme")
    if pos_data['vix'] > 30 and pos_data['is_backwardation']:
        risk_score += 10
        risk_overrides.append("VIX_Backwardation_Spike")
    if liq_data['nfl_13w_pct'] < 10:
        risk_score += 10
        risk_overrides.append("NFL_Drying_Up")
    if cov_ratio < 0.60:
        risk_score += 10
        risk_overrides.append("Low_Data_Coverage")
        
    risk_score = np.clip(risk_score, 0, 100)
    
    carry_score = None # Requires Earnings Yield from CSV
    if carry_score is not None:
        margin_score = 0.85 * margin_base + 0.15 * carry_score
    else:
        margin_score = margin_base
    margin_score = np.clip(margin_score, 0, 100)
    
    # 5. Asset Allocation (Strict Logic)
    tgt_margin = 0
    cash_min, cash_max = 15, 25
    action = "維持中性"
    
    if buy_score < 45 or risk_score > 70 or margin_score < 45 or liq_data['score'] < 30 or cov_ratio < 0.60 or modules["market_regime"] is None or modules["liquidity"] is None:
        tgt_margin = 0
        cash_min, cash_max = 30, 40
        action = "斷槓桿/禁止融資，提高現金，保留彈性"
    elif margin_score >= 80 and buy_score >= 85 and risk_score < 30 and liq_data['score'] >= 65 and cov_ratio >= 0.80 and val_score is not None:
        tgt_margin = 20
        cash_min, cash_max = 10, 10
        action = "超值區，大舉加碼"
    elif margin_score >= 70 and buy_score >= 75 and risk_score < 45 and liq_data['score'] >= 50:
        tgt_margin = 15 if val_score is not None else 5
        cash_min, cash_max = 8, 12
        action = "積極加碼"
    elif margin_score >= 60 and buy_score >= 60 and risk_score < 60 and liq_data['score'] >= 40:
        tgt_margin = 10 if val_score is not None else 5
        cash_min, cash_max = 10, 15
        action = "常規分批買入"
    elif margin_score >= 45 and buy_score >= 45 and risk_score <= 70:
        tgt_margin = 5
        cash_min, cash_max = 20, 25
        action = "買核心ETF，小量分批"

    # CC Logic
    cc_status = "HOLD_EXISTING_CC"
    if buy_score >= 75 or risk_score > 70 or cov_ratio < 0.60:
        cc_status = "STOP_NEW_CC"
    elif buy_score < 55 and margin_score < 60 and (val_score is not None and val_score < 45):
        cc_status = "NORMAL_CC"
    elif buy_score >= 45 and buy_score <= 59 and risk_score >= 60 and risk_score <= 75:
        cc_status = "ALLOW_SMALL_CC"

    # 6. JSON Output Generation
    output = {
        "signal_date": str(df.index[-1].date()),
        "generated_at": datetime.now().isoformat(),
        "data_mode": cfg.data_mode,
        "scores": {
            "Buy_Score": round(buy_score, 2),
            "Risk_Score": round(risk_score, 2),
            "Margin_Score": round(margin_score, 2)
        },
        "modules": {k: round(v, 2) if v else None for k, v in modules.items()},
        "allocation": {
            "Target_Margin_Pct": tgt_margin,
            "Target_Cash_Min_Pct": cash_min,
            "Target_Cash_Max_Pct": cash_max
        },
        "covered_call": {"status": cc_status, "reason": []},
        "action": action,
        "data_quality": {
            "coverage_ratio": round(cov_ratio, 2),
            "critical_data_ok": cov_ratio >= 0.60,
            "proxy_modules": ["market_regime_breadth"],
            "unavailable_modules": [k for k, v in modules.items() if v is None]
        },
        "risk_overrides": risk_overrides
    }
    
    with open(f"{cfg.output_directory}/latest_signals.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    
    print(f"✅ 生成完成！Buy: {buy_score:.1f} | Risk: {risk_score:.1f} | CC: {cc_status}")
    print(f"📊 Margin: {tgt_margin}% | Action: {action}")

if __name__ == "__main__":
    main()
