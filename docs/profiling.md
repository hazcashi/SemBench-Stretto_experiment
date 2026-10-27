# Profiling：推論、記憶體、搬送與缺頁

新版程式提供三種模式。**目前租用的 Vast instance 已有 instrumentation 與現場修正，直接使用 [容器手冊](vast_container_runbook.md) 的命令，不需要重新建置。** `docker/image.txt` 仍指向更早的映像；下方建置流程提供給另外建立環境者，不能將現場修正視為已發佈進舊 image。完整 GPU benchmark／trace 驗收仍待實測。

目前只跑固定 Stretto benchmark 並收集詳細 PyTorch profile（100 篇後接 728 篇）：

```bash
/opt/venv/bin/python /workspace/stretto-run/run_experiment_suite.py --profile-only
```

保留服務檢查與 8B silver reference，額外 NL smoke 預設關閉；不改原 optimizer。全程數值紀錄搭配每模型每路徑首批 trace，最多四份。此模式不包含 Nsight／CUDA UM faults，也沒有未開 profiling 的 baseline，因此 overhead 仍未量測。結果看 `/results/<run-id>/profiling/`；失敗不會自動進下一輪。

## 建置與執行

從GitHub clone到備份結果的逐步流程見 [租借GPU指南](rented_gpu_quickstart.md)。本次只改Python原始碼，該指南使用 docker/Dockerfile.source-update 沿用既有依賴，省去FlashAttention重編譯；下方主Dockerfile命令則是完整建置。

從專案根目錄執行；完整建置會編譯 FlashAttention。已有可用依賴映像時，可依 deployment 的 source-update 流程沿用環境。

```bash
docker build --platform linux/amd64 --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" -t stretto-profiling-source:local .
STRETTO_IMAGE=stretto-profiling-source:local bash scripts/reproduce.sh run
STRETTO_IMAGE=stretto-profiling-source:local bash scripts/reproduce.sh profile

# 從剛建好的新版一般映像衍生，不要使用舊 release 作為 BASE_IMAGE。
docker build --platform linux/amd64 -f docker/Dockerfile.profiling \
  --build-arg BASE_IMAGE=stretto-profiling-source:local -t stretto-nsys:local .
STRETTO_NSYS_IMAGE=stretto-nsys:local bash scripts/reproduce.sh profile-nsys
```

亦可把兩個 tag 填入 .env 的 STRETTO_IMAGE／STRETTO_NSYS_IMAGE。選配映像固定 NVIDIA 官方 CLI 2026.2.1.210，建置時記錄套件 SHA256；執行時保存 CLI 版本、environment 診斷與 image metadata。不要同時啟用 PyTorch Profiler 與 Nsight。

| 模式 | 全程資源／Linux faults | 詳細階段數據 | GPU trace |
| --- | --- | --- | --- |
| run | 有，每秒左右採樣 | 不啟用 forward／搬送詳細計時 | 無 |
| profile | 有 | 全部實際文字模型請求 | PyTorch CPU＋CUDA |
| profile-nsys | 有 | 全部實際文字模型請求 | Nsight CUDA／NVTX／OS runtime／UM faults |

缺少 Nsight CLI／必要 flags 時會在下載模型前報錯，保留 failure.txt。不自動加入 privileged、修改 driver、清除 OS page cache 或改用 Unified Memory。

## 輸出與對應

結果在 results/<run-id>/，新增 profiling/：

- small_phases.jsonl／large_phases.jsonl：原始 spans、generation、request 資源、trace 與 limitation。schema_version=1，含 role、model、request_id、batch_id、stage、phase、CPU monotonic start、wall 秒數。
- batches.csv：各 batch 的 prefill／decode GPU interval、generation CPU wall、輸出 token 數、搬送 bytes／GPU interval、顯存峰值。未量到為空值；單 token 輸出且有觀測時 decode=0。
- summary.json／summary.csv：模型、stage、phase、span 彙整；JSON 另含 request faults／peaks、資源採樣峰值與 trace coverage。
- resources.jsonl：原 run 根目錄資源檔在結束時複製到此處，保存程序與主機／cgroup 採樣。
- traces/<role>_<vanilla|kv>.json：PyTorch Chrome trace，可用 Perfetto 查看。
- nsight/*.nsys-rep、nsight/*.sqlite、nsight_environment.json、cuda_page_faults.json：Nsight trace、資料表、環境診斷與 UM 摘要。

既有 events.jsonl、telemetry_summary.json、queries.json、metrics.csv、<role>_requests.jsonl 保留。stage 區分 setup、smoke、benchmark、reference、單模型檢查；phase 從 Stretto timing context 傳入，區分 profiling、execution、prepare 等。預取執行緒的 cache span 可能沒有 batch_id，仍可由 request_id／thread_id 對應。

## Prefill／decode／compute

- cache_build_prefill：建立實體 KV cache 時對報導執行的 forward。
- vanilla_prefill：沒有預存 KV 的 generate() 第一個 forward。
- cached_query_prefill：使用預存 KV 的第一個 forward，只計實際補算部分，不宣稱重算整篇報導。
- decode：同一次 generate() 的後續 forward。首個 forward 即使輸入長度1仍是 prefill；只輸出一個 token 可沒有 decode forward。
- input_tokens：該 forward 實際 input_ids 元素數；不是原始報導總長度。
- output_tokens：實際輸出，包含首個 EOS，排除後續 padding；不用 max_new_tokens 當實測數。

保留原 generate／greedy decoding；CUDA events 在 batch／request 結尾統一讀取，不逐 token synchronize。CPU wall 是主機區間，GPU elapsed 是 stream 事件區間，可能含 launch gaps，**不是純 kernel compute time**。kernel／Memcpy 細節看代表 trace。

span 有父子包含及預取重疊，intervals_overlap=true。summary 的 sum 是工作區間總和，不能把 batch＋generation＋prefill＋decode＋load 加起來當總時間。真實總時間看 manifest／queries，Stretto 分階段時間看 component_times。

## 記憶體與搬送

約每秒記錄各程序 PID＋建立時間、角色、RSS、可讀時的 USS／PSS、faults、I/O；另記 host RAM、cgroup usage／limit、nvidia-smi 整卡顯存及 utilization。RSS 加總可能重複共享頁面；每秒採樣可能漏瞬時峰值。採樣增量不包含程序第一次／最後一次採樣之外的活動。

backend 保存 allocator allocated／reserved、階段讀值與 request peak；內層 reset 前先捕捉峰值，避免遺失搬送／組裝峰值。這不是包含 embedding 等所有程序的整卡精確連續峰值。

cache_file_load 是讀檔＋torch.load 反序列化時間，file_bytes 是檔案大小。cache_io 的 process read_bytes 是 Linux 儲存 I/O 計帳，可能包含並行預取／readahead，不能精確歸因到單一檔案；也不能保證底層是本地 SSD。page cache 命中可能 file_bytes>0、read_bytes=0。

cpu_to_cuda／cuda_to_cpu 記真正跨裝置 tensor bytes、CPU 呼叫時間與 GPU interval，同裝置 .to 不計搬送。cache_assembly 分出 padding／拼接；prefetch_wait 是等待預取；gpu_sync_wait 是原有同步點。不把整段 route time 當 PCIe bandwidth。

## 兩種缺頁與限量 trace

Linux minor／major faults 來自 /proc/<pid>/stat；request 保存程序 delta，cache load 保存 thread delta。minor faults 不代表一定讀磁碟；buffered I/O 也不必一對一產生 major faults。權限不足／程序消失與真正0次分開表示。

CUDA Unified Memory faults 由 Nsight 擷取。程式仍使用顯式 tensor 搬送，因此可能沒有 UM 事件。cuda_page_faults.json 列出 UM／page-fault tables 的 event_rows，只有資料表存在明確 fault count 欄位才另行加總；event_rows 不一定等於 faults 次數。缺少表不推定 faults=0，須搭配 CLI log／environment 判斷 collector 是否正常。見 [NVIDIA 官方文件](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)，UM faults 追蹤可能明顯增加 overhead。

每個模型只擷取 benchmark 中各路徑首個實際 batch：vanilla/direct 共用配額，KV 一份；最多四個 ranges。可能是 optimizer profiling sample，phase 會標示；排除 smoke／reference，不額外發請求補齊路徑。首批預取可能早於 batch，該段有全程 CPU span，但不保證落在代表 GPU trace 裡。

Nsight 透過 cudaProfilerStart/Stop 控制每個 backend 最多兩個 ranges。range_completed 僅表示API範圍完成，仍需檢查報告／SQLite／CLI log。結束時先通知 backend，再等待 Nsight 匯出；超過120秒會保存 timeout 限制檔。

## 驗證與 overhead

```bash
python -m pytest tests/test_profiling.py -q
# CUDA微型隨機模型，不下載權重、不呼叫OpenAI：
python -m pytest tests/test_profiling.py -k cuda_real_tiny_llama -q
```

CPU 測試涵蓋階段分類、單 token、faults／I/O 解析、權限、PID identity、trace quota、異常清理及報表。CUDA 測試比較 vanilla／cached generation，開關量測後 token 必須相同。

先跑 run，再分別跑 profile／profile-nsys。使用相同 workload／cache 狀態比較時間與結果，記錄實際 overhead；不同 run 的 KV／OS page cache 可能變 warm，新 data 目錄也不能證明全系統 cold。不自動 drop_caches。真實100篇、CUDA timing、Nsight 支援／權限與 overhead 仍待租借主機驗收。
