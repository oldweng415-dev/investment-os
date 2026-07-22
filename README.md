# Investment OS 免費資料補全：套用位置

## 直接替換的三個檔案

1. `investment_os_production.py`：取代 Repository 根目錄同名檔案。
2. `collectors/collect_free_data.py`：取代 `collectors/` 下同名檔案。
3. `.github/workflows/daily_update.yml`：用本套件的 `daily_update.yml` 取代原 Workflow。

## 主程式修改錨點

- `def load_cboe_vix3m`：放在 `def flatten_yfinance` 後、`def load_yfinance_data` 前。
- `def align_live_public_event`：放在 `def align_event_series` 後、`def apply_staleness_rules` 前。
- 完整替換 `def calculate_ai_cycle`。
- 完整替換 `def load_positioning_pit`。
- 完整替換 `def load_nowcast`。
- 完整替換 `def upcoming_events`。

## Collector 修改錨點

完整替換檔案。新版已包含：

- SEC Valuation
- FRED GDPNow
- Cboe Equity Put/Call 官方 CSV
- CFTC TFF Futures Only
- BLS／BEA／FOMC／公司財報事件
- SEC AI Cycle proxy

## Workflow 修改錨點

原本的：

- `Collect and validate free SEC valuation`
- `Commit collected valuation data`

整段改成：

- `Collect and validate free public PIT data`
- `Commit collected PIT data`

## GitHub 設定

- Secret：`FRED_API_KEY`
- Variable：`SEC_USER_AGENT`
- Optional Secret：`CFTC_APP_TOKEN`

## 首次執行預期產出

- `data/valuation_pit.csv`
- `data/macro_nowcast_pit.csv`
- `data/positioning_pit.csv`
- `data/events.csv`
- `data/ai_cycle_pit.csv`

部分外部來源暫時失敗時，Collector 會記錄 warning，主模型會透過 coverage／missing inputs 誠實反映，不會填固定 50 分。
