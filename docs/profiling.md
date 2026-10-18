# Profiling

```bash
bash scripts/reproduce.sh profile
```

此命令重跑相同 baseline，於兩個實際 `reasondb.reproduction.backend` process 的第一個 `/text_qa` HTTP request 啟用 PyTorch Profiler（CPU＋CUDA）。不是對只負責協調的 runner 做 GPU trace。預設擷取雙模型同時駐留時的已知答案 smoke inference，避免完整 benchmark trace 無限成長。trace 涵蓋推論，不含 model loading 與後續 benchmark；時間比較應使用未開 profiler 的 `run`。

- `traces/small.json`、`traces/large.json`：Chrome trace，可用 Perfetto 查看。CUDA kernel、Memcpy、GPU idle gaps 的可見性以實際 CUPTI/driver 權限為準。
- 若 profiler 初始化/匯出失敗，保存 `<role>_limitation.txt`；不能用空 trace 宣稱已取得 GPU 證據。不透過無限制 privileged 修正。
- `<role>_requests.jsonl`：endpoint、HTTP status、revision、request elapsed、allocated/reserved/peak allocated bytes、profile flag。這是單一 backend process 的記憶體，不是全 GPU 峰值。
- `resources.jsonl`：每秒 GPU 使用量及整個 runner process tree 的 RSS。GPU memory 來自 nvidia-smi，可能漏掉一秒內的瞬時峰值，不等同 CUDA allocator 計數。
- `events.jsonl`、`telemetry_summary.json`：沿用 Stretto 的 query/operator/optimizer/backend event schema。啟動／下載／cache 準備與 query wall time 分開看。
- `queries.json`、`metrics.csv`：query output cardinality、CostSummary component_times、optimizer guarantees、執行計畫與 silver precision/recall/F1。

`prefill_s`、`decode_s` 目前為 null，並未可靠地拆開測量；token counts、CPU→GPU transfer 分相也不捏造。可在 trace 中進一步分析 generate kernels。PyTorch Profiler 的 record_shapes/profile_memory、每秒 nvidia-smi、HTTP 逐筆 JSONL 均有 overhead，manifest 保存啟用狀態與採樣頻率，尚未量測相對 overhead。

純 query pipeline 使用 upstream runtime profiling 重新測量新模型的 latency/observations；TEXT_SPECS 的 quality/fake_cost 只是 bootstrap ranking，不是新模型實測品質。
