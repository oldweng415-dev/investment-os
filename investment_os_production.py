import yfinance as yf
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import json
from datetime import datetime, timedelta

def main():
    print("啟動 Investment OS 實戰運算引擎...")
    
    # --- 1. 定義時間區間 (取過去 5 年數據做百分位數計算) ---
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 5)
    
    # --- 2. 獲取市場數據 (Market Data) ---
    print("正在抓取 yfinance 市場數據...")
    tickers = ["SPY", "QQQ", "SOXX", "NVDA", "^VIX"]
    df_mkt = yf.download(tickers, start=start_date, end=end_date)['Close']
    df_mkt.ffill(inplace=True)

    # --- 3. 獲取總經與流動性數據 (FRED) ---
    print("正在抓取 FRED 總經與流動性數據...")
    # BAMLH0A0HYM2: 高收益債利差 (信用壓力)
    # WALCL: 聯準會總資產 (流動性指標)
    # T10Y2Y: 10年與2年期美債利差 (衰退指標)
    # ICSA: 初次請領失業救濟金人數 (就業指標)
    fred_series = ['BAMLH0A0HYM2', 'WALCL', 'T10Y2Y', 'ICSA']
    df_fred = web.DataReader(fred_series, 'fred', start_date, end_date)
    df_fred.ffill(inplace=True)

    # 數據合併與清理
    df = df_mkt.join(df_fred, how='outer').ffill().dropna()
    today = df.iloc[-1]

    # 百分位數計算輔助函數
    def calc_percentile(series, val):
        return (series < val).mean() * 100

    # =========================================
    # --- 模組 1: Market Regime (市場狀態) 21% ---
    # =========================================
    spy_200 = df['SPY'].rolling(200).mean().iloc[-1]
    qqq_200 = df['QQQ'].rolling(200).mean().iloc[-1]
    spy_trend = min(100, max(0, 50 + (today['SPY'] / spy_200 - 1) * 500))
    qqq_trend = min(100, max(0, 50 + (today['QQQ'] / qqq_200 - 1) * 500))
    hyg_oas_pct = 100 - calc_percentile(df['BAMLH0A0HYM2'], today['BAMLH0A0HYM2']) # 利差越大分數越低
    MR = 0.4 * spy_trend + 0.4 * qqq_trend + 0.2 * hyg_oas_pct

    # =========================================
    # --- 模組 2: Macro (總體經濟) 17% ---
    # =========================================
    # 殖利率曲線倒掛解除通常是衰退開始，這裡簡化為長短天期利差相對位階
    spread_pct = calc_percentile(df['T10Y2Y'], today['T10Y2Y']) 
    # 失業救濟金 4週均線與26週均線比較
    claims_4w = df['ICSA'].rolling(4).mean().iloc[-1]
    claims_26w = df['ICSA'].rolling(26).mean().iloc[-1]
    claims_ratio = claims_4w / claims_26w if claims_26w > 0 else 1
    claims_score = 100 - min(100, max(0, (claims_ratio - 0.9) * 500)) # 失業增加則扣分
    MAC = 0.5 * spread_pct + 0.5 * claims_score

    # =========================================
    # --- 模組 3: Liquidity (流動性) 19% ---
    # =========================================
    # 聯準會資產負債表 13 週 (約65個交易日) 變化
    walcl_13w = df['WALCL'].pct_change(65).iloc[-1] 
    LIQ = min(100, max(0, 50 + walcl_13w * 1500))

    # =========================================
    # --- 模組 4: Positioning (市場部位/恐慌) 17% ---
    # =========================================
    # 實戰中缺少期權資料，以 VIX 百分位數作為恐慌買盤指標 (VIX 越高，反向作多分數越高)
    POS = calc_percentile(df['^VIX'], today['^VIX']) 

    # =========================================
    # --- 模組 5: Valuation (估值) 12% ---
    # =========================================
    # 實戰 API 無前瞻本益比，以 QQQ 乖離 5 年均線作為估值位階 Proxy
    qqq_5y = df['QQQ'].rolling(252*5).mean().iloc[-1]
    VAL = 100 - min(100, max(0, (today['QQQ'] / qqq_5y - 1) * 100))

    # =========================================
    # --- 模組 6: AI Cycle (AI 週期) 14% ---
    # =========================================
    # 無法自動化抓財報，暫以 NVDA 相對 SPY 252日強弱勢作為 Proxy
    nvda_rs = df['NVDA'].pct_change(252).iloc[-1] - df['SPY'].pct_change(252).iloc[-1]
    AI = min(100, max(0, 50 + nvda_rs * 100))

    # =========================================
    # --- 計算三大決策分數 ---
    # =========================================
    # 修正項：恐慌加分與過熱扣分
    B_panic = 10 if (POS >= 80 and today['BAMLH0A0HYM2'] < 5) else 0 
    P_euphoria = 10 if (POS <= 20 and VAL <= 40) else 0

    Buy_Score = (0.25*MR + 0.15*AI + 0.20*VAL + 0.15*LIQ + 0.10*MAC + 0.15*POS) + B_panic - P_euphoria
    Buy_Score = max(0, min(100, Buy_Score))

    X_penalty = 10 if today['BAMLH0A0HYM2'] > df['BAMLH0A0HYM2'].quantile(0.9) else 0
    Risk_Score = 100 - (0.28*MR + 0.22*LIQ + 0.18*MAC + 0.12*POS + 0.10*VAL + 0.10*AI) + X_penalty
    Risk_Score = max(0, min(100, Risk_Score))

    Carry = 50 # 實質利差 Proxy，此處簡化設定為中性
    Margin_Score = 0.35*Buy_Score + 0.35*(100-Risk_Score) + 0.15*LIQ + 0.15*Carry
    Margin_Score = max(0, min(100, Margin_Score))

    # =========================================
    # --- 資金管理與動作判定 (五階層設計) ---
    # =========================================
    target_margin = 0
    target_cash = 0
    action_text = ""

    if Risk_Score > 70:
        target_margin = 0
        target_cash = 30
        action_text = "Risk > 70: 禁止融資，強制保留現金 (停損/不賣 Covered Call)"
    elif Buy_Score < 45 or Risk_Score > 75:
        target_margin = 0
        target_cash = 30
        action_text = "不加碼，優先保留彈性 (現金 30-40%)"
    elif 45 <= Buy_Score <= 59 and 60 <= Risk_Score <= 75:
        target_margin = 5
        target_cash = 20
        action_text = "只買核心 ETF，小量分批 (融資上限 5%)"
    elif 60 <= Buy_Score <= 74 and 45 <= Risk_Score <= 60:
        target_margin = 10
        target_cash = 10
        action_text = "常規分批買入 (融資 5-10%)"
    elif 75 <= Buy_Score <= 84 and 30 <= Risk_Score <= 45:
        target_margin = 15
        target_cash = 10
        action_text = "積極加碼 (融資 10-15%)"
    elif Buy_Score >= 85 and Risk_Score < 30:
        target_margin = 20
        target_cash = 10
        action_text = "恐慌/超值區，滿載槓桿 (融資上限 20%)"
    else:
        target_margin = 5
        target_cash = 15
        action_text = "持有核心部位，維持中性狀態"

    # =========================================
    # --- 輸出結果寫入 JSON ---
    # =========================================
    output = {
        "date": end_date.strftime("%Y-%m-%d"),
        "scores": {
            "Buy_Score": round(Buy_Score, 2),
            "Risk_Score": round(Risk_Score, 2),
            "Margin_Score": round(Margin_Score, 2)
        },
        "allocation": {
            "Target_Margin_Pct": target_margin,
            "Target_Cash_Pct": target_cash,
        },
        "action": action_text,
        "modules": {
            "Market_Regime": round(MR, 2),
            "Macro": round(MAC, 2),
            "Liquidity": round(LIQ, 2),
            "Positioning": round(POS, 2),
            "Valuation": round(VAL, 2),
            "AI_Cycle": round(AI, 2)
        }
    }

    # 儲存為 JSON 供前端與 GitHub Action 抓取
    with open('latest_signals.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    
    print(f"\n--- [決策完成] ---")
    print(f"Buy: {round(Buy_Score, 1)} | Risk: {round(Risk_Score, 1)}")
    print(f"建議動作: {action_text}")
    print(f"資料已成功寫入 latest_signals.json")

if __name__ == "__main__":
    main()