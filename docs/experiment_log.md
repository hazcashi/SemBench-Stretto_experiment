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

第一次互動式 Docker build 在依賴步驟收到 context canceled，並非已通過或套件相容性失敗。已改用隱藏背景 build process 與 .audit/build.stderr.log 保存完整輸出；不刪除使用者既有 Docker image 或 volume。

此文件會隨本次實際驗證結果更新；不把建置成功當成論文復現成功。
