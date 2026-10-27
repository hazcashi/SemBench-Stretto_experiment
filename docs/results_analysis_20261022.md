# 2026-10-22 實驗分析入口

本機資料已移至專案內 `stretto-results/results/`。分析完全離線，不呼叫模型、OpenAI 或遠端 GPU，也不改原始結果。

- [瀏覽器圖表報告](../stretto-results/analysis/report.html)：單一 HTML，內嵌 SVG，可離線開啟。
- [Markdown 完整報告](../stretto-results/analysis/report.md)：可在 VS Code 用 Markdown Preview 閱讀。
- [圖表 PDF](../stretto-results/analysis/charts.pdf)：10 張圖表，另有各張 PNG／SVG。
- [分析數據 JSON](../stretto-results/analysis/analysis_summary.json)、[總覽 CSV](../stretto-results/analysis/overview.csv)、[答案差異 CSV](../stretto-results/analysis/answer_differences.csv)。

| 指標 | 100 篇 | 728 篇 |
|---|---:|---:|
| 球員—報導結果列數 | 655 | 4,699 |
| Benchmark 查詢 | 4.65 分鐘 | 21.41 分鐘 |
| 其中正式 execution | 2.08 分鐘 | 15.13 分鐘 |
| 整輪，含 8B silver reference 與服務檢查 | 8.48 分鐘 | 41.72 分鐘 |
| 整卡顯存採樣峰值 | 23.32 GiB | 23.35 GiB |
| 程序 RSS 加總採樣峰值，含載入期 | 14.29 GiB | 17.12 GiB |
| 可見 cgroup 使用峰值，包含 cache 等 | 27.59 GiB | 42.00 GiB |
| 與 8B silver 的列級 F1 | 0.9618 | 0.9666 |

主要結果：正式 execution 實際使用 3B vanilla 抽 points、8B vanilla 抽 assists；兩輪都沒有正式 KV 呼叫。728 篇候選採樣中，KV 每 item 時間分別是 vanilla 的約 1.80 倍（3B）、1.50 倍（8B）。KV cache 準備約 203 秒，占查詢時間 15.8%，有供候選採樣使用，但最終大量推論沒有選 KV。

728 篇正式執行的 prefill／decode GPU event intervals 分別合計 494.93／325.13 秒；顯式 H2D＋D2H 區間合計 1.93 秒。這支持優先分析 forward／生成路徑，但 GPU events 內含 launch gaps，不能稱為純計算量或直接判定 compute-bound。代表 trace 顯示 kernels 間有空隙；沒有 CPU 使用率與未開 profiling 的 baseline，尚不能定論空隙原因或整場 idle 比例。

兩輪 KV 檔案載入區間的程序 `read_bytes` 都為 0，不能將反序列化／讀檔時間稱為實體 SSD 傳輸時間。100 篇重用既有 cache、728 篇實際新建 cache，總時間不是公平 cold/warm 對照。答案差異以 name＋game_id 對齊核對：100 篇 points 不同 25 列，728 篇不同 157 列，兩輪 assists 都相同；8B silver 不是人工 ground truth。

## 重建分析

使用已含 matplotlib、numpy 的 Python：

```bash
python scripts/analyze_profile_results.py \
  --results-root stretto-results/results \
  --output stretto-results/analysis
```

這台電腦也可沿用已存在的 `stretto-profile:local` Docker image；在專案根目錄 PowerShell：

```powershell
New-Item -ItemType Directory -Force -Path stretto-results/analysis | Out-Null
$repo = (Get-Location).Path.Replace('\','/')
docker run --rm --entrypoint /opt/venv/bin/python --mount "type=bind,source=$repo,target=/project,readonly" --mount "type=bind,source=$repo/stretto-results/analysis,target=/analysis" stretto-profile:local /project/scripts/analyze_profile_results.py --results-root /project/stretto-results/results --output /analysis
```

原始結果以唯讀方式掛載；只有 analysis 目錄可寫。腳本核對答案鍵值、列級一致性、時間分解及 GPU trace 時間聯集；輸入 SHA256 與先前遠端清單比對，記於 `analysis_inputs.json`。詳細區間、request、階段記憶體、Linux faults 與 trace 統計均另存 CSV，避免只留下圖片。2026-10-27 依使用者換機需求，分析報告與原始結果 ZIP 已加入 Git 白名單；解壓後的巨大 JSONL 仍排除，請依 [換機還原說明](../stretto-results/README.md) 操作。

完整方法、限制與後續實驗建議在圖表報告中；本次沒有 Nsight Unified Memory faults、重複試驗、未開 profiling 的同條件 baseline，因此不報 profiling overhead、信賴區間或與論文的 speedup。
