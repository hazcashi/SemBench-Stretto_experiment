# 實驗結果與換機還原

2026-10-22 的完整結果已隨 Git 保存，不需要 Git LFS、不需要連回租借 GPU，也不需要下載模型才能閱讀報告。

- [圖表報告 HTML](analysis/report.html)：下載後以瀏覽器開啟；GitHub 網頁本身只顯示原始碼。
- [完整中文報告 Markdown](analysis/report.md)：可直接在 GitHub 或 VS Code 預覽。
- [10 頁圖表 PDF](analysis/charts.pdf)：另有每張圖的 PNG／SVG。
- [原始結果 ZIP](raw-results-20261022.zip)：249 個檔案，原始大小 378,616,879 bytes，壓縮後約 29.3 MB。
- [封存清單與 SHA256](archive-manifest.json)、[原遠端逐檔 SHA256](remote-inventory.json)、[下載驗證紀錄](download-verification.json)。

ZIP 完整保存成功與失敗兩輪先前嘗試、成功 100／728 profile、所有 trace、raw JSONL、資源採樣、輸出 CSV、執行設定與 logs。分析目錄另存衍生數據、答案差異及分析輸入 hashes。舊驗證紀錄內的 Downloads／遠端路徑是歷史證據；目前還原位置如下。

## 回自己的電腦

已有此 repository，先在專案根目錄執行：

```bash
git pull --ff-only
```

尚未下載 repository：

```bash
git clone https://github.com/hazcashi/SemBench-Stretto_experiment.git
cd SemBench-Stretto_experiment
```

只看分析不需要 Python 或 Docker，直接開 `stretto-results/analysis/report.html` 或 PDF。

若要讀原始 JSONL／trace 或重新產生圖表，在專案根目錄使用 Python 3.9 以上（無額外套件）：

```bash
python scripts/restore_results.py
```

此指令先檢查 ZIP SHA256，再還原到 `stretto-results/results/`，逐檔核對原遠端 SHA256；已有相同檔案就略過，遇到不同檔案會停止而不覆寫。也可只驗證已還原的檔案：

```bash
python scripts/restore_results.py --verify-only
```

Windows 沒有 Python 時，可在專案根目錄 PowerShell 使用內建解壓指令（目標 results 目錄應尚未存在）：

```powershell
Expand-Archive -LiteralPath .\stretto-results\raw-results-20261022.zip -DestinationPath .\stretto-results
```

兩個成功結果資料夾：

| 輪次 | 目錄 |
|---|---|
| 100 篇 profile | `results/20261022T093856Z-9f1aa934` |
| 728 篇 profile | `results/20261022T094727Z-ab3bbe42` |
| 成功批次設定／進度 | `results/suites/20261022T093839Z-7a6ccad7` |

`.env`、API token、SSH 私鑰、模型權重與可重新生成的 KV cache 沒有上傳。要重跑推論，需在自己的電腦另行設定憑證；此備份足以閱讀及重新分析現有結果。新程式已提交並不代表舊 GHCR image 自動更新，後續部署要另行建置／發佈。
