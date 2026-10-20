# 實作與驗證紀錄

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
