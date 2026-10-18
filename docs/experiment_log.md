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

執行中／待確認：

- 本地完整映像建置與 FlashAttention GPU kernel smoke；toolbox／一般 CUDA tensor smoke 已通過。
- GHCR workflow 執行與最終 image digest（未成功前不提供虛構 pull 位址）。
- HF gated access / OpenAI 小額實際請求：使用者尚未提供憑證，未執行。
- 48GB GPU：個別模型、雙模型共存、實際 query／benchmark、GPU trace，皆尚未驗證。

第一次互動式 Docker build 在依賴步驟收到 context canceled，並非已通過或套件相容性失敗。後續建置保留 .audit/build.stderr.log 與 inference-build.stderr.log。CPU/GPU 前置驗證完成後，本機磁碟剩約 12GiB，因此停止本地完整建置，改由 GitHub Actions 完成；已確認 Docker buildx 子程序與編譯程序也停止，不只停止父 CLI。

完整 FA2 編譯包含大量不需要的 backward kernels；改用 upstream 提供的 FLASHATTENTION_DISABLE_BACKWARD 選項建置推論映像，保留全部 forward head dimensions。尚未完成的完整 image／FA kernel 驗證仍須以實際 CI 與 GPU 輸出為準。

本地暫存清理：只按已核對的 25 個 cache ID 刪除本次建置 cache（20.79GB），並移除本次下載的 CUDA devel image／暫用 dependencies image；既有 SonarQube container 與四個 volumes 保留。Docker 停止後使用 Windows Virtual Disk API 離線壓縮 Docker data VHDX，回傳成功，檔案由約 50.8GB 縮至約 10.9GB；Docker 已重啟並核對原 container／volumes，C 槽約 43GiB 可用。

此文件會隨本次實際驗證結果更新；不把建置成功當成論文復現成功。

租借後 `doctor` 已補上真正的 CUDA tensor 及 FlashAttention BF16/head-dim 128 kernel smoke，通過才進入 HF／API／模型下載階段；這項最終映像 kernel 檢查仍待 48GB GPU 執行，未以本地前置 tensor smoke 代替。

服務重啟補查：在 Linux container 重現 server 主動關閉連線後的 `TIME_WAIT`，未設 `SO_REUSEADDR` 的埠檢查回傳 EADDRINUSE，即使沒有服務在 listen。Linux probe 已改為與 HTTP server 相同的重用選項；Windows 保留原行為，避免平台語意差異。新增回歸測試驗證同一埠能重新啟動，既有占用埠拒絕測試也保留。輕量 Linux/Python3.12 測試結果為 **7 passed、3 skipped**（未安裝 torch/OpenAI/HF）；完整依賴版本需在最終映像中跑滿 10 項。此修正會在首次完整建置成功後，以相同已驗證依賴層發佈新版程式映像。
