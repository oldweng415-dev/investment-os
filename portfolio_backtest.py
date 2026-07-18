# 在您的 investment_os_production.py 中加入此段
import firebase_admin
from firebase_admin import credentials, firestore

# 1. 下載您的 serviceAccountKey.json (在 Firebase 控制台 -> 專案設定 -> 服務帳戶 -> 產生私鑰)
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def update_dashboard(data):
    # 2. 將計算出的決策分數上傳至 Firestore
    doc_ref = db.collection('artifacts').document(appId).collection('public').document('data').collection('latest_signals').document('latest_signals')
    doc_ref.set({
        'buy': round(data['buy'], 2),
        'risk': round(data['risk'], 2),
        'margin': round(data['margin'], 2),
        'action': data['action'],
        'timestamp': firestore.SERVER_TIMESTAMP
    })
    print("✅ Dashboard 資料已同步至 Firebase Firestore")
```

### 第三步：前端託管部署 (React 儀表板)
您現在可以直接將我們之前寫的 `App.jsx` 部署到 Firebase Hosting。

1.  **安裝 Firebase CLI**：在您的電腦終端機輸入 `npm install -g firebase-tools`。
2.  **初始化部署**：在專案資料夾執行 `firebase init`。
    *   選擇 `Hosting`。
    *   選擇您的 Firebase 專案 ID。
    *   公開目錄設為 `dist` 或 `build`。
3.  **進行構建與上傳**：
    *   `npm run build`
    *   `firebase deploy`

### 第四步：自動化排程 (讓它自己跑)
不用每天手動點開電腦，我們利用 **GitHub Actions** 讓它在雲端自動執行：

1.  在您的 GitHub 儲存庫建立 `.github/workflows/daily_update.yml`。
2.  填入排程規則：

```yaml
name: Daily Investment OS Update
on:
  schedule:
    - cron: '30 22 * * *' # 每天台灣時間 06:30 AM (收盤後)
jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install yfinance pandas firebase-admin
      - name: Run OS Engine
        env:
          FIREBASE_CREDENTIALS: ${{ secrets.FIREBASE_CREDENTIALS }}
        run: python investment_os_production.py
```

### 總結執行路徑
*   **數據端**：Python 腳本每日盤後在 GitHub Actions 雲端自動執行 -> 計算結果 -> `firebase_admin` 上傳 Firestore。
*   **視覺端**：您的 React 網頁掛在 Firebase Hosting 上，隨時監控 Firestore 的變化。
*   **體驗端**：您只需在手機開網頁，就能看到 Python 系統幫您算出「今日該加碼還是減碼」。

這是一個標準的**金融數據流自動化架構**。現在，您只要把 `firebaseConfig` 填入您的 React 程式，您的這套 OS 就正式進入「全自動化操盤助手」的層級了。有哪一個步驟需要我細寫嗎？