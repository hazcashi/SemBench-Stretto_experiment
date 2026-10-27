# 實作與驗證紀錄

## 2026-10-27：GitHub 換機備份

- 使用者要求將程式與完整結果上傳 GitHub，方便回自己的電腦 clone／pull。保存全部 249 個原始檔案（378,616,879 bytes），包括先前失敗紀錄，壓縮為 `stretto-results/raw-results-20261022.zip`（29,341,933 bytes）；分析 HTML／Markdown、10 張 PNG／SVG、PDF 與衍生 CSV／JSON 直接保存。
- ZIP SHA256：`355fc07fa6039c11b15aeeea2733ec72c5a1f26c19b6cd3501b36dd9b5788530`。已檢查 ZIP CRC 及所有原始檔案 SHA256，與 `remote-inventory.json` 一致。敏感憑證掃描沒有命中；`.env`、私鑰、模型與 KV cache 未納入。
- 新增 `scripts/restore_results.py`，只依賴 Python 標準函式庫，檢查封存雜湊及路徑、拒絕覆寫不一致檔案、逐檔驗證還原內容。已在空白目錄實際還原 249 個檔案，第二次 verify-only 通過且沒有重寫檔案。
- `.gitignore` 只對本次封存／分析檔案開放白名單，解壓後大型原始結果仍排除。程式修正與啟動腳本一併保存；此操作不建置 Docker image、不執行 GPU 或付費 API 實驗。

## 2026-10-22：離線分析與圖表

- 本機資料已由 Downloads 移至專案內 `stretto-results/results`；新增 `scripts/analyze_profile_results.py`，輸出自包含 HTML、Markdown、10 張 PNG／SVG、PDF 與衍生 CSV／JSON，入口為 [結果分析](results_analysis_20261022.md)。原始資料不修改、不啟動遠端／模型／API。
- 以原始 events、requests、phase spans 分離 benchmark execution／optimizer profiling／reference；以 phase 父子關係避免重複加總，GPU trace 取區間聯集，Linux faults 僅加採樣 delta。保留資料口徑與 profiler 開銷未知的限制。
- 正式 execution 兩輪都沒有 KV 呼叫；728 篇 cache preparation 約 203 秒，候選 KV/vanilla 每 item 時間比約 1.80（3B）／1.50（8B）。728 篇 execution 907.63 秒，benchmark query 1,284.69 秒，整輪含 reference 2,502.91 秒。
- 逐欄對齊原始 CSV：100 篇 655 列中 points 不同 25 列；728 篇 4,699 列中不同 157 列；assists 皆相同。與既有 silver F1 相符；不是人工正確率。這是資料觀察，並未變更 optimizer 或宣稱取得加速。

## 2026-10-22：100／728 profile 完成與還機前備份核對

- suite `20261022T093839Z-7a6ccad7` 全部 completed；100 篇結果 `20261022T093856Z-9f1aa934`、728 篇結果 `20261022T094727Z-ab3bbe42`。兩份 manifest 都 completed、backend_errors 為空且沒有 failure.txt；遠端沒有執行中的實驗程序。
- 本機 `C:/Users/yungchang/Downloads/stretto-results/results` 與遠端 `/results` 逐檔比對：249 個檔案、378,616,879 bytes，檔案清單、大小與 SHA256 全部一致；沒有漏檔或讀取錯誤。核對證據保存在本機同層 `remote-inventory.json` 與 `download-verification.json`。
- 兩輪結果內 JSON／JSONL 全部可解析，benchmark／silver CSV 分別各 655／4,699 列；profiling batches 分別 3,436／19,612 列。每輪四份 PyTorch trace 全數存在且包含實際 CUDA kernel 與 GPU memcpy 事件，共八份；prefill／decode、搬送 bytes 與 allocator peak 欄位都有數值紀錄。這是檔案完整性與基本量測可用性驗收，尚未完成瓶頸／品質／量測語意的詳細分析。
- 已具備在本機分析這兩輪所需的結果，可結束本次租用；未代使用者刪除 instance。未保存可重新產生的模型權重／KV cache。Nsight Unified Memory faults 與無 profiling baseline／overhead 本次未量測，不屬於漏下載。

## 2026-10-22：額外 NL smoke 超出上游 optimizer 支援範圍

- suite `20261022T091624Z-dc11f2d6` 的 100-run `/results/20261022T091641Z-95e3367a` 在 smoke 失敗，尚未執行固定 benchmark。GPT 產生抽取勝隊、join、依抽取欄位 filter、project 的計畫，最終觸發 `CascadesSelectivities.get_inter_selectivity` 的 `assert level == 0`。
- 本機 `base_optimizer.py`／`gd_optimizer.py` SHA256 分別為 `7863a71562010e9304ad913a1da2a01d687832b01a5f6f275b414e5e12da1ff2`／`6089946ac322be82be2d9655bb78223b943cac06b3d49e78fd3e24cbba6d60bf`，與來源清單的上游比對值一致。optimizer 只模擬 level 0，不可刪除 assert 或把 level 0 的比例套給後續步驟。
- 使用者選擇優先執行原固定 benchmark，將額外 NL smoke 改選跑。runner／helper 新增 `--nl-smoke`（預設關閉），manifest 記錄選擇；保留失敗即停止，不把 NL smoke 記成已通過。不變更 benchmark 問題、資料、模型、optimizer、cache／generate 行為或 silver reference。
- 使用者進一步要求只跑 benchmark 並搭配詳細 profile。suite 新增 `--profile-only`：inspect／doctor／tiny CUDA → 100 profile → 728 profile；不執行 baseline run。全程數值紀錄與限量 PyTorch trace 保留，Nsight／UM faults 與 overhead 仍 pending。
- 本機容器離線驗證 **58 passed、1 skipped**（CUDA 測試本機跳過）。新測試覆蓋預設略過 NL、明確選跑、NL 失敗清理／停止、固定 benchmark＋silver 路由、100/728 模式選擇；使用上游真實固定 LogicalPlan 與 DependencyGraph 驗證 points／assists 是兩條 level 0 cascade。沒有呼叫付費 API／模型；不宣稱此測試替代物理計畫生成與完整 GPU 實驗。
- 遠端已更新 runner、兩個 launcher，SHA256 檢查通過；原檔在 `/workspace/stretto-run/patch-20261022-benchmark/original`，紀錄在 `/results/host-info/benchmark-entry-fix.json`。遠端入口測試 4 passed、批次腳本測試 8 passed（全部離線），並驗證已安裝 CLI 的 `--profile-only` 與階段順序。GHCR image 尚未重建；權重、cache 與兩次失敗結果保留。沒有啟動實驗，由使用者執行。

## 2026-10-22：首輪 GPU 執行與跨表 filter 修正

- 使用者啟動 suite `20261022T085110Z-a612b659`。doctor 完成，tiny CUDA 為 1 passed／20 deselected，run directory 為 `/results/20261022T085132Z-c7b2d7af`。3B、8B 個別與並行 smoke 完成，模型權重已下載；run manifest status=failed，總時間約 319.95 秒，cache_bytes_after_run=2127115552。
- 100-run 在 NL smoke 規劃階段因 DuckDB BinderException 中止，尚未執行正式 benchmark、100-profile 或 728。TraditionalFilter 將同一虛擬輸入內、來自不同實體表的欄位全部套用最後一個 input_table，將抽取結果欄位錯誤指向 teams_to_games。
- 改為逐欄保留 ConcreteColumn binding；virtual SQL 使用 logical column alias。SqlQuery 表格重新命名時同步改動 binding，不改寫 SQL 字串常數、不修改共用 observation。沒有更改模型、query、資料、生成參數或跳過 LLM。
- 新增 DuckDB CPU 回歸測試：原 image 可重現同類 BinderException，修正後本機完整測試 52 passed／1 skipped；遠端針對此 bug 的 4 個 CPU 測試通過，不呼叫模型或外部 API。
- 已備份並部署兩份來源修正至目前 instance，原檔位於 `/workspace/stretto-run/patch-20261022-filter/original`；`/results/host-info/filter-column-fix.json` 保存前後 SHA256。啟動器新增這兩份來源 hash 供後續結果對齊。GHCR image 尚未重建。
- 保留失敗結果與所有模型／cache，沒有自行重跑實驗。使用者重跑原批次命令會建立新的 suite；完整流程與品質驗收仍 pending。LLM 產生的規劃語意仍需以後續結果檢查，這次修正不宣稱改善規劃品質。

## 2026-10-22：Vast 預建映像與批次交接

- 使用者選定直接使用 GHCR 預建 Stretto 映像，不在 NVIDIA CUDA 基礎模板現場編譯。操作入口為 `docs/vast_container_runbook.md`。
- 重新查驗固定 tag `436207bd211943709930577ec44895d8a1db0cd5`：index digest `sha256:afe378f49f356403d2769c6d49e81cb9beb859773817765125da16b6841bbb68`，存在 linux/amd64 manifest。
- 新增 `scripts/run_experiment_suite.py`，依序 inspect／doctor／tiny CUDA／100 run／100 profile／728 run／728 profile。每階段獨立 log，退出碼與結果 manifest 檢查失敗即停止；suite.json 保存進度。啟動器與批次 CPU 測試共 12 passed，GPU 實驗未執行。
- 已 SSH 登入 instance 57182443，確認 RTX 6000 Ada 49140MiB、檔案系統約 274GiB，當時為 CUDA 基礎映像，沒有 `/opt/stretto` runtime。使用者明確確認無檔案需保留並授權更新。
- update_template 與 recycle API 都回覆 success，目標為上述 Stretto tag，onstart 改為 true。重建後舊 Direct `89.221.67.144:16394` 與 Proxy `ssh4.vast.ai:18248` 拒絕連線；新容器啟動、最新 SSH endpoint 尚待確認。
- 使用者最新要求改為自行執行實驗。因此只繼續環境確認與準備；沒有執行 doctor、模型下載、API 推論、run 或 profile，沒有上傳 `.env` 或批次腳本。批次腳本仍在本機，需於新容器可連線後上傳。

### 重建後環境與交接完成

- 使用新 Direct SSH `root@89.221.67.144:16234` 成功登入。確認 `/opt/stretto/source`、`/opt/venv` 已存在，Python 3.12.3／torch 2.9.1+cu128／FlashAttention 2.8.3.post2 可 import；CUDA available=True、device_count=1、compute capability=(8,9)，RTX 6000 Ada 49140MiB、driver 580.178.04。
- 已上傳 `reproduce_in_container.py`、`run_experiment_suite.py` 與本機 `.env` 至 `/workspace/stretto-run`，目錄權限 700、憑證檔權限 600。僅解析憑證格式，未輸出內容，未驗證外部 API 權限。
- 遠端 `inspect` 成功顯示 100 reports／27 operators，紀錄在 `/results/host-info/preparation-inspect.log`；兩份腳本語法檢查通過，tmux 已存在。
- 實驗由使用者啟動。沒有執行 doctor、CUDA kernel 測試、模型下載、run 或 profile；HF/OpenAI 權限、GPU 推論、100/728 實驗及 trace 均 pending。

## 2026-10-18

已完成：

- 寫入 reproduction_plan 後才開始修改程式；保留 upstream 預設入口。
- 修改前逐檔比對 upstream、記錄 source_provenance，兩 submodule 固定 upstream 指定 SHA；外層 Git 正確登錄 gitlinks。
- Windows palimpzest 尾隨句點路徑使首次 checkout 失敗；針對此 submodule 的一次 checkout 關閉 NTFS 路徑檢查並使用 sparse-checkout，固定 SHA 已成功。不修改 global Git 設定。正式 Linux recursive clone 不受此問題影響。
- `python -m pytest tests/test_reproduction.py -q`：初輪 6 passed、1 skipped；新增 failure manifest 與 missing-credentials 測試後為 6 passed、3 skipped。本機未安裝 torch/OpenAI/HF；完整 9 項於容器驗證。
- `python -m compileall -q stretto-main/stretto-main/reasondb/reproduction scripts/fetch_rotowire.py`：通過。
- `python scripts/fetch_rotowire.py --destination .audit/rotowire-fetch`：五份公開資料從固定 commit 下載並通過 SHA256，資料不加入 Git。
- 在 Linux CUDA base container 以 `bash -n` 檢查 reproduce.sh：通過。
- Linux/Python3.12 使用 uv 0.8.22 解析成功：143 個鎖定依賴，生成含 hashes 的 requirements.lock。
- CUDA devel/runtime base 固定 digest；FA2 sdist 固定 SHA256，無 torch2.9 對應 wheel，因此從 source 編譯。
- 程式已推送 [GitHub 專案](https://github.com/hazcashi/SemBench-Stretto_experiment)。首次 push 被 scanner 將 upstream 註解的模型類別名誤認為 Mistral key；核對並改寫該註解後成功，未繞過保護規則。
- 已透過既有 GitHub 登入手動 dispatch 映像 workflow。既有 PAT 沒有 packages scope，因此由 workflow 的 GITHUB_TOKEN 執行 GHCR publish。
- 使用本次建立的 Linux/Python3.12 dependencies container，安裝 pinned kvpress 並對目前原始碼執行 root 測試：**9 passed in 7.94s**，無 skip。真實工具箱只含兩個指定文字模型，upstream 預設工具箱仍包含原模型。
- 同一組已鎖定依賴的 GPU container：torch 2.9.1+cu128、CUDA 12.8、`torch.cuda.is_available()=True`；RTX 4060 Laptop 的小型 CUDA tensor 計算通過。此驗證不代表 48GB 雙模型驗收。
- DuckDB 1.3.0 `INSTALL vss; LOAD vss;` 實測通過；原始 `KvTextQaModelWrapper` 與 modified kvpress imports 通過。

尚待確認：

- 最終映像的 FlashAttention GPU kernel smoke；CPU 匯入／toolbox 已通過，一般 CUDA tensor 在本機前置容器通過。
- HF gated access / OpenAI 小額實際請求：使用者尚未提供憑證，未執行。
- 48GB GPU：個別模型、雙模型共存、實際 query／benchmark、GPU trace，皆尚未驗證。

第一次互動式 Docker build 在依賴步驟收到 context canceled，並非已通過或套件相容性失敗。後續建置保留 .audit/build.stderr.log 與 inference-build.stderr.log。CPU/GPU 前置驗證完成後，本機磁碟剩約 12GiB，因此停止本地完整建置，改由 GitHub Actions 完成；已確認 Docker buildx 子程序與編譯程序也停止，不只停止父 CLI。

完整 FA2 編譯包含大量不需要的 backward kernels；改用 upstream 提供的 FLASHATTENTION_DISABLE_BACKWARD 選項建置推論映像，保留全部 forward head dimensions。完整 image／FA 匯入已由 CI 通過；GPU kernel 驗證仍須以租借主機輸出為準。

本地暫存清理：只按已核對的 25 個 cache ID 刪除本次建置 cache（20.79GB），並移除本次下載的 CUDA devel image／暫用 dependencies image；既有 SonarQube container 與四個 volumes 保留。Docker 停止後使用 Windows Virtual Disk API 離線壓縮 Docker data VHDX，回傳成功，檔案由約 50.8GB 縮至約 10.9GB；Docker 已重啟並核對原 container／volumes，C 槽約 43GiB 可用。

此文件會隨本次實際驗證結果更新；不把建置成功當成論文復現成功。

租借後 `doctor` 已補上真正的 CUDA tensor 及 FlashAttention BF16/head-dim 128 kernel smoke，通過才進入 HF／API／模型下載階段；這項最終映像 kernel 檢查仍待 48GB GPU 執行，未以本地前置 tensor smoke 代替。

服務重啟補查：在 Linux container 重現 server 主動關閉連線後的 `TIME_WAIT`，未設 `SO_REUSEADDR` 的埠檢查回傳 EADDRINUSE，即使沒有服務在 listen。Linux probe 已改為與 HTTP server 相同的重用選項；Windows 保留原行為，避免平台語意差異。新增回歸測試驗證同一埠能重新啟動，既有占用埠拒絕測試也保留。輕量 Linux/Python3.12 測試為 **7 passed、3 skipped**（未安裝 torch/OpenAI/HF）；修正版最終映像則以完整依賴跑滿 **10 passed in 4.47s**。

### 最終打包驗證

- [完整建置 CI](https://github.com/hazcashi/SemBench-Stretto_experiment/actions/runs/39120539396) 成功：Python3.12／CUDA12.8.1／torch2.9.1+cu128／FA2 2.8.3.post2，FA source 編譯約 7,201 秒；`pip check`、DuckDB vss、FA／kvpress 匯入、真實工具箱與當時的 9 項測試通過。
- [修正版 CI](https://github.com/hazcashi/SemBench-Stretto_experiment/actions/runs/39129558557) 成功：核對基底 revision 與依賴輸入未變，保留同一已編譯環境，加入 Linux 重啟修正及 GPU preflight；資料 SHA、依賴／匯入／工具箱及 **10 項測試全部通過，無 skip**。
- 最終 source revision：`54a648938ee92c243fa4f5fdabd8cf1ae53b9ab3`。固定映像：`ghcr.io/hazcashi/sembench-stretto_experiment@sha256:5ce3088106b16f1108ef70944e2cb779ee71289310e4a0c1e0dedc9e0d0710b7`。
- 匿名 GHCR token／manifest 請求成功，SHA256 與 Linux/amd64 平台已核對；未使用 GitHub／Docker 登入憑證。壓縮 layers 合計 7,456,465,085 bytes，解壓空間另計。
- `docker/image.txt` 為啟動器預設；`docker/release.json` 保存來源、依賴基底、版本與 CI 證據。程式已打包完成，**尚未完成模型／API 存取及 48GB GPU 實驗驗收，也不宣稱論文復現完成**。

## 2026-10-20：詳細 profiling 實作（借用電腦）

- 新增可停用的 forward prefill/decode、cache build prefill、檔案載入／反序列化、H2D/D2H、cache assembly、prefetch/sync wait 記錄；不替換 generate 或改變推論參數。
- 新增 host/cgroup RAM、程序 RSS/USS/PSS、Linux minor/major faults 與 I/O、request/batch allocator peaks；GPU 無法量測不填假0。
- 新增 profiling JSONL/CSV/summary；PyTorch trace 改為正式 benchmark 每模型／路徑首個 batch，檢查匯出是否含 GPU events。Nsight 使用獨立映像與 profile-nsys，保存 UM fault 報告；不自動要求 privileged。
- 本機系統 Python 是3.10.6；其 hashlib.file_digest 不支援造成既有 subset 測試失敗，因此另下載官方免安裝 Python3.12.10 到忽略的 .audit/python312，沒有更改系統 Python。pytest/psutil/requests 安裝於忽略的專案 .venv。
- 已下載並 SHA256 校驗五份公開 Rotowire CSV，以執行既有子集測試；檔案在忽略的 benchmarks/files，不加入Git。
- Python3.12.10 執行根目錄 tests：26 passed, 5 skipped（Windows TIME_WAIT一項；缺少torch/OpenAI/HF相關三項；CUDA微型模型一項）。新版 profiling 的CPU測試20項通過。
- Git Bash 的 bash -n scripts/reproduce.sh、Python compileall、git diff --check 通過。
- Docker daemon 在這台借用電腦不可連線，沒有建置或發佈新版 image；docker/image.txt 與 release.json 保留原發佈紀錄。Dockerfile.profiling 的實際套件安裝也尚未驗收。
- 待租借主機：CUDA微型隨機模型開關量測之token一致性、實際100篇查詢、有效GPU trace、Nsight權限與UM collector、三種模式overhead比較。

### GitHub原始碼發布前檢查

- 新增 docs/rented_gpu_quickstart.md，包含 GitHub clone、沿用依賴建置新版程式映像、HF/API設定、CUDA微型測試、run/profile/profile-nsys與結果備份；README改為引導新版映像。
- 發現原 smoke 題目要求 Yes/No，而 upstream boolean endpoint 明確要求1/0；已對齊為驗證已知真值1，加入6個CPU回歸案例，錯誤/不確定回答仍失敗。
- Python3.12根目錄測試：32 passed, 5 skipped in 6.33s。Bash語法及git diff --check通過。
- 此次發布範圍是原始碼/文件，不代表新版Docker映像已建置或48GB GPU已驗收。
