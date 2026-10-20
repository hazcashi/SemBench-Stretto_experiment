# 單卡文字復現計畫與驗證清單

## 已查證的現況（2026-10-18）

原始碼位於 `stretto-main/stretto-main/`。外層 Git repository 尚無 HEAD，下載原始碼沒有自己的 Git metadata；不可將最新 upstream SHA 當成下載版本的來源證明。

架構：`RaccoonDB` → `Executor.setup/prepare` → `SelfCorrectionReasoner` / `PlanConfigurator` → `GradientDescentOptimizer` → physical operators → `KvTextQABackend` → localhost Flask KV server → modified kvpress / Transformers / PyTorch。

- `reasondb/config/model_registry.py:7`：8B 原為 small、70B 為 large；無 3B。
- `scripts/start_servers_text.sh:44`：預設多 GPU，啟動 8B、70B、影像與文字 embedding。
- `reasondb/interface/default_operator_toolbox.py:104`：模型、quality、fake_cost、vanilla 與 compression 綁定，不可僅改 registry。
- `reasondb/interface/config.py:35`：GPT4o 負責 configurator 與 reasoner；`reasondb/database/database.py:84` 另使用 GPT4oMini 與 OpenAI text embeddings 建立 metadata，因此雲端依賴不只 planner。
- `reasondb/backends/kv_cache_text_qa_server.py:1344`：BF16、device_map=auto，先嘗試 FlashAttention 2，失敗可 fallback。CLI 從 registry 取得模型 choices。
- `reasondb/coordinator/capabilities.py:36`：text readiness 仍要求影像 embedding；baseline 專用入口不能直接套用該啟動設定。
- `scripts/generate_kv_cache.py:293`：支援 limit，但必須與實際執行子集完全一致，否則 prepare 仍會補建 cache。
- `reasondb/evaluation/benchmarks/rotowire.py:34`：原固定查詢集目前只有一個 query（兩個 joins、points/assists extracts）；不得宣稱為多查詢 sweep。
- `reasondb/backends/inference_stats.py` 與 `reasondb/monitor/` 已有 request、GPU、operator、query telemetry；保留原輸出再補 run manifest。

## 實作設計

新增顯式 single_gpu profile 與 runner，沿用原 Executor、reasoner、optimizer、KV prepare/inference；不改 upstream 預設，不使用 replay 冒充執行。工具箱新增可選 specs/modality 參數以避免建立任何影像 operator。3B 新 registry entry 不改既有 8B 的 upstream size_class，profile 內角色另行定義。

首 100 個 reports 依 CSV 原順序固定選擇，關聯表以 game_id 過濾，players/teams 保留原順序且只保留 referenced names；manifest 保存來源/子集 hash。最小 query 使用 reports 文字 filter；之後執行原 Rotowire 固定 query。precision/recall=0.9，兩模型各 cr=0.8＋vanilla，expected_attention、實體 cache、single request concurrency。新模型採 upstream quality/cost 作 bootstrap priors，明示非測量值；runtime profiler 必須重新執行。

## 階段與驗收

### 2026-10-20 詳細 profiling 擴充

- [x] 加入可停用的 prefill/decode forward 計時、cache load/H2D/D2H/assembly/wait spans。
- [x] 加入程序 RSS/USS/PSS、host/cgroup RAM、Linux faults/I/O、跨 reset 顯存峰值保存。
- [x] 將限量 trace 移至正式 benchmark 首個實際 batch；區分 stage/phase，CPU-only trace 不當成 GPU 成功。
- [x] 加入 profile-nsys、獨立 Dockerfile.profiling、UM fault SQLite 摘要與報表匯出等待。
- [x] CPU 數據契約測試、Python/Bash 語法檢查；詳細結果見 experiment_log。
- [ ] 新版一般映像及選配 Nsight 映像建置與發佈；舊 release digest 保持不變。
- [ ] CUDA 微型模型 token 一致性、48GB GPU 真實100篇 workload、Nsight collector/UM faults 支援、overhead 比較。

- [x] 讀取 AGENTS.md 與主要原始碼。
- [x] 比對 upstream tree，固定 upstream/submodules，保存 provenance。
- [x] 建立 profile、文字工具箱、deterministic subset、runner 與失敗紀錄。
- [x] 建立 Dockerfile、doctor/run/profile/access 入口與 GHCR workflow；完整映像建置驗收另列。
- [x] CPU：語法、依賴、registry/profile 路由、無影像 operator、子集關聯、readiness/cleanup／Linux 埠重啟、匯入；最終映像內 10 tests 通過。
- [x] 完整 Docker image build、FlashAttention 編譯／匯入、公開 GHCR 發佈與匿名 digest 驗證；固定版本見 `docker/release.json`。
- [ ] 最終映像的 CUDA／FlashAttention GPU kernel smoke（一般 CUDA tensor 已在本機前置容器通過）；由租借後的 doctor 執行。
- [ ] 有權限後：HF 兩模型 config/tokenizer 存取，OpenAI 實際小額請求。
- [ ] 48GB GPU：nvidia-smi / torch CUDA；3B 單獨、8B 單獨、雙模型；query；固定 benchmark；backend trace。

## 資源預算與風險

BF16 11B 權重約 22GB（20.5GiB），另計 CUDA contexts、KV、activation、embedding 與 allocator。48GB 是目標而非已測保證；先限 batch，記錄實際峰值，OOM 必須失敗不靜默量化或 offload。建議 RAM 64GB、磁碟至少 100GiB 可用給最小部署，完整 sweep 可能數百 GB。權重下載約 22GB 加 cache；100 篇 cr=0.8 KV 的真實大小由 token 長度與模型層數決定，量測後才擴大。

本機 RTX 4060 Laptop 8GiB、driver 592.82，不能驗收雙模型；初始 C 槽約 62GiB free，本地前置檢查後改由 GitHub Actions 完成完整建置。本次暫存清理與磁碟狀態見 experiment_log；未刪除既有 Docker container／volumes。模型權限與 API key 尚未提供。GHCR image 已建成並通過匿名 manifest 驗證；實際 GPU kernel／模型／實驗仍未驗收。

### 固定工作量的本地實測（非模型推論結果）

依已固定的五份 CSV，首 100 篇 reports 產生：275 players、30 teams、655 players_to_games、183 teams_to_games。reports 共 171,666 字元，平均 1,716.66，最長 3,347；尚未取得 gated tokenizer 權限，因此不把字元數當作實測 token 數。

HF 公開 metadata 回報 8B checkpoint 有 8,030,261,248 個 BF16 parameters，3B 有 3,212,749,824；權重理論合計約 22.49GB／20.94GiB，仍未包含 KV、CUDA contexts、activation 與 embedding。磁碟與 VRAM 峰值以租借主機上的實際紀錄為準。
