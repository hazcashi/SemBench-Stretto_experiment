# Vast 操作手冊：Rotowire 728 篇／60 題

更新：2026-10-27。這份是本次實驗唯一操作手冊。

本次範圍為全部 Rotowire 五張資料表、728 篇報導，以及 Stretto `RotowireRandom` 上游產題器的 60 題。模型為 Llama-3.2-3B／Llama-3.1-8B，單張約 48GB NVIDIA GPU。這是 Rotowire 實驗，不是 Movies／Email 或整篇論文的全部實驗。

## 1. 租用 Vast

- GPU：單張約 48GB NVIDIA GPU。
- 磁碟：建議配置約 300GB；程式會檢查至少 100GiB 可用空間。
- RAM：建議至少 64GB。
- 模式：SSH container／Interactive shell server。
- Image：使用下面已發佈的預建環境。

```text
ghcr.io/hazcashi/sembench-stretto_experiment@sha256:5ce3088106b16f1108ef70944e2cb779ee71289310e4a0c1e0dedc9e0d0710b7
```

若 Vast 模板只接受 tag，環境 tag 為：

```text
ghcr.io/hazcashi/sembench-stretto_experiment:436207bd211943709930577ec44895d8a1db0cd5
```

此 image 提供 Python／PyTorch／CUDA／FlashAttention／kvpress；接下來從 GitHub 取得本次程式並執行。不要把 image 內的舊 source 當作最新程式，也不用在 Vast 容器裡再啟動 Docker 或重編 FlashAttention。

先取得兩個 Llama 模型的 Hugging Face gated access，並準備可用的 `HF_TOKEN` 與 `OPENAI_API_KEY`。完整推論與 8B reference 需要時間；不要用先前單題的時間估算整輪 60 題。

## 2. 上傳 .env（本機 PowerShell）

在本機 repository 根目錄準備 `.env`，至少有：

```dotenv
HF_TOKEN=你的_HuggingFace_token
OPENAI_API_KEY=你的_OpenAI_key
```

`.env` 不進 GitHub。以下 `PORT`、`USER`、`HOST` 換成 Vast 顯示的 SSH 資訊；如果使用專用私鑰，ssh／scp 另外加 `-i "你的私鑰路徑"`。

```powershell
ssh -p PORT USER@HOST "mkdir -p /workspace/stretto-run && chmod 700 /workspace/stretto-run"
scp -P PORT .env USER@HOST:/workspace/stretto-run/.env
ssh -p PORT USER@HOST
```

下方步驟都在遠端 Bash 執行。

## 3. 檢查環境、建立 tmux（遠端）

```bash
nvidia-smi
/opt/venv/bin/python --version
chmod 600 /workspace/stretto-run/.env
```

必須能找到 `/opt/venv/bin/python`，且 GPU 是租用的那張卡。若找不到 Python，先核對 Vast image。

Vast 通常以 root 登入。缺少 git／tmux 時只補操作工具：

```bash
apt-get update
apt-get install -y --no-install-recommends git tmux
```

若目前已在 tmux，可以沿用；否則執行：

```bash
tmux new -A -s stretto60
```

## 4. 從 GitHub 取得程式與資料

第一次使用這台機器：

```bash
cd /workspace
git clone --recurse-submodules https://github.com/hazcashi/SemBench-Stretto_experiment.git
cd /workspace/SemBench-Stretto_experiment
```

如果這台機器已經 clone，改用：

```bash
cd /workspace/SemBench-Stretto_experiment
git pull --ff-only
git submodule update --init --recursive
```

下載本次所需的全部五張 Rotowire 表，並核對固定上游版本的 SHA256：

```bash
/opt/venv/bin/python scripts/fetch_rotowire.py
```

這一步只下載公開資料；兩個模型的權重會在正式啟動時從 Hugging Face 下載至 `/data/hf`。

## 5. 建立本次 source，執行離線檢查

仍在 `/workspace/SemBench-Stretto_experiment` 執行：

```bash
/opt/venv/bin/python scripts/package_full_rotowire.py
/opt/venv/bin/python -m zipfile -e dist/stretto-full-rotowire.zip dist
/opt/venv/bin/python dist/stretto-full-rotowire/scripts/run_full_rotowire.py --check
```

ZIP 在 Vast 上由目前 checkout 產生，不必從本機上傳。它保存程式與資料 hash，獨立 source 使用預建環境的依賴；不覆寫 `/opt/stretto/source`，也不打包 `.env` 或模型權重。

`--check` 不讀憑證、不呼叫 API、不下載模型。看到 `Offline checks passed` 才繼續。離線測試使用合成 fixture 檢查產題／接續功能；正式產題會使用實際 8B 模型的 filter 結果。

## 6. 正式啟動：這一行

在 tmux 中執行：

```bash
/opt/venv/bin/python /workspace/SemBench-Stretto_experiment/dist/stretto-full-rotowire/scripts/run_full_rotowire.py --env-file /workspace/stretto-run/.env
```

自動執行順序：

1. CUDA／FlashAttention、磁碟、HF／OpenAI 權限檢查。
2. 3B、8B 各自回應檢查，以及兩模型同卡回應檢查。
3. 用真實 8B filter 推論建立 overlap stats，再依上游 generator 固定 60 題。
4. 每題執行 Stretto global optimization、真實本地推論、8B vanilla reference 與評分。
5. 保存每題 plan、階段時間、backend requests、資源採樣及有限量 GPU trace。

終端會印出 `Experiment: ...`、`Results: ...`，以及 `[1/60]` 到 `[60/60]`。過程會呼叫 OpenAI API，需有額度。

本次固定設定：BF16、batch=1、sample=100、precision=recall=0.9；兩模型各有 cr=0.8 KV 與 vanilla 候選，使用 expected_attention。Optimizer 自行選擇，沒有強制使用 KV。題庫使用本次 8B 結果產生，不能宣稱逐題等同原論文 70B 產生的題庫。品質指標是對 8B silver 的一致性，不是人工標註正確率。

預設使用 `profile`：全程保存 timing／資源採樣；每模型每路徑最多擷取首批代表 PyTorch CUDA trace，合計最多四份。不是整輪無限 kernel trace，也不含 Nsight Unified Memory faults。

## 7. 斷線、失敗與接續

離開 tmux：按 `Ctrl+B`、放開，再按 `D`。重新 SSH 後回去：

```bash
tmux attach -t stretto60
```

若程式失敗，在相同程式版本下，重新執行第 6 節的同一行。已完成 reference／評分且 CSV hash 通過的題目會略過；失敗或尚未評分的題目會重跑。不要在同一張 GPU 同時啟動兩輪。

若要全部重跑，明確指定新目錄：

```bash
/opt/venv/bin/python /workspace/SemBench-Stretto_experiment/dist/stretto-full-rotowire/scripts/run_full_rotowire.py --env-file /workspace/stretto-run/.env --experiment-dir /results/full-rotowire60/repeat-02
```

若更新 GitHub 程式，先停止目前實驗，再重新執行第 4 節的 pull 與第 5 節；新 source hash 會使用新實驗目錄，避免混用不同版本的 checkpoint。

## 8. 結果在哪、怎樣才算完成

- `/results/full-rotowire60/<hash>-profile/state/progress.json`：進度及最後一個 run。
- 同一個 `state/queries.json`：固定的 60 題與 logical plans。
- 同一個 `state/preparation.json`：產題前置工作／cache 準備時間。
- `state/query-000/` 至 `state/query-059/`：benchmark／silver CSV、metrics、plan 與 hash checkpoint。
- `/results/<run-id>/manifest.json`、`workload_summary.json`、`metrics.csv`：本次嘗試的狀態和逐題結果。
- `/results/<run-id>/profiling/`：GPU／記憶體／forward timing／trace；`per_query_spans.csv` 按題目彙總。
- `/results/<run-id>/services/`：模型服務 logs；失敗另有 `failure.txt`。

把終端的 run ID 代入下方 `RUN_ID` 檢查：

```bash
cat /results/RUN_ID/workload_summary.json
cat /results/RUN_ID/manifest.json
```

必須是 `status: completed`、`scored_queries: 60`，且沒有 backend errors，才算流程完成。品質是否達標另看逐題指標。接續的結果會標記 `resumed_queries`、`source_run_id`，不能把最後一次嘗試的時間當作全 60 題總時間。Profiling spans 有巢狀／重疊，也不能全部直接相加。

## 9. 關機前備份

遠端打包整個 `/results`，保留所有嘗試與 checkpoints：

```bash
tar -czf /workspace/stretto-run/rotowire60-results.tar.gz -C /results .
```

本機 PowerShell 下載：

```powershell
scp -P PORT USER@HOST:/workspace/stretto-run/rotowire60-results.tar.gz .
```

確認備份完整後再 destroy instance。`/data` 下的模型／KV 不加入 Git，Vast 容器內檔案也不保證 destroy 後保留。

## 驗證狀態

程式曾通過 Linux／Python 3.12 CPU 全回歸 `70 passed, 1 skipped`（CUDA），以及執行包 `--check`。本手冊整併後另驗證路徑／打包／離線測試。真實 60 題 GPU、API、品質及完整 trace 仍需在本次租借主機驗收；沒有宣稱已跑完。
