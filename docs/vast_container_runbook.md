# Vast Stretto SSH 容器：連線、檢查、實驗與備份

> **2026-10-27 更新：100／728 profile 已成功完成並備份至 Git。** 回自己電腦閱讀／還原請看 [結果與換機說明](../stretto-results/README.md)。下方「未上傳 GitHub／尚待實驗」敘述屬當時交接紀錄；來源修正現已納入 Git，但舊 GHCR image 尚未重新建置，兩者不可混為一談。

> **目前採用的流程：Vast 拉取已建好的 Stretto image → SSH 登入 → 一個指令依序跑完 100 篇與 728 篇。** 不需要重新安裝 Python/PyTorch 或編譯 FlashAttention。若目前 instance 是 `vastai/base-image`，它還不是 Stretto 環境；不要在裡面執行下方實驗，也不要改走現場編譯。

## 最短操作流程：預建 image＋一次跑完

**目前建議指令（2026-10-22，取代下方較早的實驗交接狀態）：** 預設不再執行額外的「主隊獲勝」自然語言 smoke。只執行上游 `ROTOWIRE_QUERIES` 固定計畫、原 optimizer／模型推論與 8B vanilla silver reference；OpenAI 仍用於 operator 設定。下列命令依序跑 **100 篇 profile → 728 篇 profile**，包含環境／微型 CUDA／模型服務檢查，不跑未開 profiling 的 baseline 輪次：

```bash
# 在遠端 tmux 裡執行；已在 tmux 就不用再開一層。
/opt/venv/bin/python /workspace/stretto-run/run_experiment_suite.py --profile-only
```

若只想先跑 100 篇一輪：

```bash
/opt/venv/bin/python /workspace/stretto-run/reproduce_in_container.py profile
```

`profile` 保存全程階段／資源數據，PyTorch trace 只擷取正式 benchmark 每模型每路徑首批，最多四份；**不是全程 kernel trace，也不含 Nsight Unified Memory faults**。輸出在 `/results/<run-id>/profiling/`，批次狀態在終端顯示的 `/results/suites/<suite-id>/suite.json`。只有 `status: completed` 表示流程完成；precision/recall、trace coverage／limitations 另需驗收。不跑 baseline 時無法從這兩輪直接計算 profiling overhead。

第二次失敗 `20261022T091624Z-dc11f2d6` 發生於額外 NL smoke 產生的 extract→filter 相依計畫；上游 GD optimizer 的 selectivity 僅支援 level 0。兩個 optimizer 檔案 SHA256 與 `source_provenance.json` 記錄一致，沒有刪 assertion 或捏造 selectivity。新增離線測試確認：固定 Rotowire 的 points／assists 被依賴分析分成兩條 level 0 cascade；預設不呼叫 NL smoke，benchmark 與 silver 仍執行。**這不代表完整 GPU benchmark 已跑通。** 如需研究原先的 NL smoke，可明確加 `--nl-smoke` 至單輪 helper 的 `run/profile` 命令；它仍可能產生不支援的計畫，失敗會保留並停止，不建議目前啟用。

此更新已部署 runner、兩個啟動腳本與測試；本機 58 passed／1 skipped，遠端 12 個離線回歸測試通過。遠端備份位於 `/workspace/stretto-run/patch-20261022-benchmark/original`，變更紀錄在 `/results/host-info/benchmark-entry-fix.json`。GitHub／舊 GHCR image 尚未包含這些未發佈修正；目前 instance 使用已部署的修正版，不要 recycle。下方為保留的初始部署與其他模式說明。

**最新修正（2026-10-22，優先於下方原先交接紀錄）：** 首批 `20261022T085110Z-a612b659` 的 doctor 與 tiny CUDA 已通過，100-run 在 natural-language smoke 規劃的 TraditionalFilter 發生跨表欄位 BinderException。已在本機與目前 instance 修正 `reasondb/database/sql.py`、`reasondb/operators/filter/traditional_filter.py`，保留逐欄實體來源及虛擬欄位別名；本機 52 passed／1 skipped，遠端 CPU 回歸 4 passed。原檔備份在 `/workspace/stretto-run/patch-20261022-filter/original`，變更 hash 在 `/results/host-info/filter-column-fix.json`；下次啟動器也會記錄這兩個檔案的 hash。**沒有重啟實驗；由使用者在 tmux 重跑同一批次命令。** 權重與 KV cache 保留，會建立新的結果目錄；不需重新租機器或編譯。GHCR 舊 image 尚未包含此修正，重建／recycle 會失去這次遠端修正。完整 100/728 與 profile 尚未完成。

**本次交接狀態（2026-10-22）：** instance 57182443 已完成預建環境更換，新 Direct SSH 是 `root@89.221.67.144`、port **16234**。已登入確認 Python 3.12.3、PyTorch 2.9.1+cu128、FlashAttention 2.8.3.post2、CUDA 可見單張 RTX 6000 Ada 49140MiB、約 274GiB 檔案系統容量。兩個啟動腳本與 `.env` 已上傳到 `/workspace/stretto-run`；`.env` 權限 600，僅驗證格式，不輸出憑證。`inspect` 成功顯示 100 篇與 27 operators。**尚未執行 doctor、下載模型、呼叫 HF/OpenAI 或啟動實驗；由使用者自行啟動。** 不需再執行 A、C 節的換 image／上傳步驟，也不要再次 recycle。

這台機器現在可直接使用以下 **本機 PowerShell** 連線：

```powershell
ssh -i "$env:USERPROFILE/.ssh/id_ed25519_stretto_vast" -o IdentitiesOnly=yes -p 16234 root@89.221.67.144
```

登入後，若已在 tmux（`echo "$TMUX"` 有值）直接執行下方 Python；否則先單獨執行 `tmux new -A -s stretto`，進入 tmux 後再貼以下 **遠端 Bash** 指令：

```bash
/opt/venv/bin/python /workspace/stretto-run/run_experiment_suite.py
```

此指令會開始模型下載與有費用的 API 呼叫，依序完成檢查、100 篇 run/profile、728 篇 run/profile。各輪 GPU 推論與品質是否通過，以實際結果為準；目前尚未跑過。執行詳解見 D 節，完成後跳第 8 節備份。

### A. 使用已發佈的 image

Vast 自訂模板的 **Image Path:Tag** 填入下方固定值，Launch Mode 選 **Interactive shell server, SSH**，Disk Space 約 **300GB**。用乾淨的自訂模板；不要保留 NVIDIA CUDA 模板啟動 Portal 等應用的 onstart 指令。由 Vast 自動下載、啟動 image。Python、PyTorch、FlashAttention、Stretto、固定版 Rotowire CSV 都已在 image 裡；模型權重首次 run 才下載。

```text
ghcr.io/hazcashi/sembench-stretto_experiment:436207bd211943709930577ec44895d8a1db0cd5
```

這是 GitHub Container Registry（GHCR）的預建映像；`git clone` 只取得原始碼。一般 Vast instance 已是容器，不能在 NVIDIA CUDA 容器內再執行普通 `docker run` 啟動它。現有 CUDA instance 不會因為修改帳號模板而自動換成 Stretto，但官方 CLI 提供 `vastai update instance ID --image IMAGE --onstart true` 更新現有 instance 的映像，無需先 destroy 再搶 GPU。先備份／確認容器內無需保留資料，且需具備 API 更新權限；更新後重新核對 SSH endpoint 與 runtime。如果 API 已換成目標 image、實際仍跑舊容器，需 `vastai recycle instance ID` 原地重建（會清除容器檔案，保留 GPU 優先權）。2026-10-22 使用者確認沒有需保留資料後，instance 57182443 的 update 與 recycle API 均回覆 success；等待新環境驗收，不能把 API success 當成實驗完成。[官方 CLI 原始碼](https://github.com/vast-ai/vast-cli/blob/master/vast.py)／[模板設定](https://docs.vast.ai/guides/templates/template-settings)／[Docker-in-Docker 限制](https://docs.vast.ai/guides/reference/faq/instances)

### B. Open 與 SSH 是不同入口

`Open` 進入的 **Instance Portal** 是網頁應用入口，本實驗不用啟動 Syncthing 或 TensorBoard。回到 Vast **Instances** 列表，在該 instance 卡片按小型 **`>_` SSH 圖示**，複製 Terminal Connections 的 Direct SSH（必要時 Proxy SSH）。用此指令更新 VS Code Remote SSH 的 HostName／Port，保留專用 IdentityFile。不要用 Portal 的網頁連線埠當 SSH port。[官方 SSH 說明](https://docs.vast.ai/guides/instances/connect/ssh)／[Portal 說明](https://docs.vast.ai/guides/instances/connect/instance-portal)

### C. 上傳兩個小型啟動工具與憑證

兩個工具只是批次執行／SSH 環境處理，不重建推論環境。它們尚未包含於上方已發佈的固定 image，所以上傳本機檔案。以下在 **本機 PowerShell**，替換新 instance 的 `PORT`、`USER`、`HOST`：

```powershell
Set-Location C:\Users\yungchang\Desktop\lin_chen\SemBench-Stretto_experiment
ssh -i "$env:USERPROFILE/.ssh/id_ed25519_stretto_vast" -o IdentitiesOnly=yes -p PORT USER@HOST "mkdir -p /workspace/stretto-run && chmod 700 /workspace/stretto-run"
scp -i "$env:USERPROFILE/.ssh/id_ed25519_stretto_vast" -o IdentitiesOnly=yes -P PORT scripts/reproduce_in_container.py scripts/run_experiment_suite.py .env USER@HOST:/workspace/stretto-run/
ssh -i "$env:USERPROFILE/.ssh/id_ed25519_stretto_vast" -o IdentitiesOnly=yes -p PORT USER@HOST
```

### D. 一次啟動，不用等 100 篇完成再手動按下一輪

在 **遠端 Bash**：

```bash
ls -l /opt/venv/bin/python /opt/stretto/source/reasondb/reproduction/runner.py
nvidia-smi
chmod 600 /workspace/stretto-run/.env
command -v tmux
```

前兩個檔案不存在就先停：目前不是我們的 image。若缺 tmux，只安裝此操作工具：`apt-get update && apt-get install -y tmux`。若已在 tmux（`echo "$TMUX"` 有值），直接執行下面 Python；否則先 `tmux new -s stretto`。

```bash
/opt/venv/bin/python /workspace/stretto-run/run_experiment_suite.py
```

執行順序固定為 **inspect → doctor → tiny CUDA 測試 → 100 篇 run → 100 篇 profile → 728 篇 run → 728 篇 profile**。同一張 GPU 依序跑，沿用下載好的模型，分開保存每輪結果；不清除 OS page cache。每輪先檢查退出碼與 manifest，失敗會自動停止，不帶著錯誤跑下一輪；各輪 doctor 仍檢查至少 100GiB 可用磁碟。不額外等待人工確認品質，品質數值留待整批分析；執行成功不等於達到目標 precision/recall。

終端一開始顯示 `Suite directory: /results/suites/...`。該目錄的 **suite.json** 記錄進度、每階段 log 與對應結果目錄；只有 `status: completed` 才表示全批完成。失敗保留已有結果；重跑此指令會開始新批次，不會自動跳過先前階段。

可 `Ctrl+B` 然後 `D` 離開 tmux，重連後 `tmux attach -t stretto`。全部完成後依本頁第 8 節備份整個 `/results`。300GB 不保證足以存全部 KV／trace，需依實際用量判斷；中途空間不足會停，勿把配額檢查關掉。

這裡的「正式」指本專案 **728 篇完整 Rotowire、同一固定查詢**，不是論文全部實驗。批次涵蓋 PyTorch profiling 與 Linux faults；目前預建一般 image 不含 Nsight CLI，所以這一鍵流程**不含 CUDA Unified Memory faults**。run/profile 的 cache 狀態不同時，不能直接相減宣稱 profiling overhead。

本機驗證：啟動器與批次共 12 項 CPU 測試通過，包括非零退出、缺少結果、失敗 manifest／backend errors 時停止後續工作與保留紀錄。已重新查驗 GHCR 固定 tag 的 linux/amd64 manifest；新批次工具尚未推送或包入這個既有 image，本次 instance 已透過 SCP 上傳。遠端環境 import、CUDA 枚舉與 inspect 通過，GPU kernel／完整推論與整批實驗仍 pending。

以下保留逐步執行、結果解讀與備份的詳細參考；使用上述批次後不需要再重跑第 3～7 節。

更新日期：2026-10-22。適用於一般 Vast Docker instance，模板直接使用：

```text
ghcr.io/hazcashi/sembench-stretto_experiment:436207bd211943709930577ec44895d8a1db0cd5
```

Launch Mode 為 **Interactive shell server, SSH**。Vast 在外層啟動映像；登入後已在容器內，不需要 Docker daemon。不要在這裡執行 VM 手冊的 `docker pull`、`docker run`、`scripts/reproduce.sh`。模型、資料、查詢與推論設定沿用映像；新 helper 只處理 SSH 環境、憑證與啟動方式。

**建立 instance 時，Disk Space 請配置約 300GB，不能沿用 16GB 等小容量預設值。** GPU 方案顯示的主機剩餘磁碟，不等於租給你的容量；建立後看 instance 的 Disk 分母。16GB 不足以容納目前模型，且無法通過 100GiB free 檢查。Vast 官方說明 container disk 建立後無法調整，需另建足夠容量的 instance；不以主機 `df` 可能顯示更大磁碟作為突破租用額度的依據。[儲存規則](https://docs.vast.ai/guides/instances/storage/types)

本機驗證：啟動器的 6 項 CPU 測試通過；在既有映像模擬 SSH 缺少 PATH／錯誤 HF_HOME 後，inspect 成功顯示 3B＋8B、batch=1、100 篇與 27 個 operators。Bash／PowerShell 指令與內嵌 Python 語法也已檢查。Vast SSH 啟動、GPU kernel、雙模型、正式 benchmark 和 trace 仍需在租借 instance 驗收；這些本機檢查沒有呼叫 HF/OpenAI 或下載權重。

Vast SSH 模式會取代 image ENTRYPOINT，SSH session 也可能缺少原環境變數，因此固定使用 `/opt/venv/bin/python` 與本手冊的 helper。[Vast Docker 說明](https://docs.vast.ai/guides/instances/docker-environment)、[自訂模板說明](https://docs.vast.ai/guides/templates/template-settings)

## 1. 先登入，確認租到的是正確環境

SSH 與 VS Code Remote SSH 設定沿用 [VM 手冊第 1 節](vast_vm_runbook.md#1-從本機-ssh-登入與-vs-code-remote-ssh)。使用自己的專用私鑰，`USER`、`HOST`、`PORT` 依 Vast 的連線指令替換。以下 **本機 PowerShell**：

```powershell
ssh -i "$env:USERPROFILE/.ssh/id_ed25519_stretto_vast" -o IdentitiesOnly=yes -p PORT USER@HOST
```

若遠端拒絕公鑰，而且 Connect 視窗只有 Direct／Proxy 指令，該視窗沒有金鑰編輯功能。可使用官方 attach SSH API 補加到一般容器。先在 Vast Keys → API Keys 新增臨時 key，授權 instance 管理／附加 SSH key，再於**本機 PowerShell**執行（`INSTANCE_ID` 是租用 instance ID，不是 host 或 machine ID）：

```powershell
Set-Location C:\Users\yungchang\Desktop\lin_chen\SemBench-Stretto_experiment
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\attach_vast_ssh.ps1 -InstanceId INSTANCE_ID
```

API key 只貼到命令啟動後的隱藏輸入提示；不貼聊天或指令參數。工具只送出專用 `.pub` 公鑰，不停止／重建機器，也不把 API key 存到檔案。顯示 attached successfully 後再試 SSH；完成用途後可撤銷這個臨時 API key。[官方 API](https://docs.vast.ai/api-reference/instances/attach-ssh-key)

登入後，在 **遠端 Bash** 逐項執行。此階段不下載權重、不呼叫 OpenAI：

```bash
uname -m
nvidia-smi
free -h
df -h / /workspace /dev/shm
ls -l /opt/venv/bin/python /opt/stretto/configs/single_gpu.json /opt/stretto/source/reasondb/reproduction/runner.py
/opt/venv/bin/python -c 'import sys, torch, flash_attn; print("Python:", sys.version); print("Torch:", torch.__version__, "CUDA runtime:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available()); print("Visible GPUs:", torch.cuda.device_count())'
```

預期 `x86_64`、上述檔案存在、CUDA available=True、租用單卡時 Visible GPUs=1。`free -h` 可能顯示整台主機 RAM，不能當成此 instance 的額度；仍需對照 Vast 分配值與後續資源紀錄的 cgroup limit。`/dev/shm` 容量也要保留紀錄，無法在容器內透過修改 `docker run` 任意擴大。

若 `/workspace` 不存在，先執行 `mkdir -p /workspace`，再重跑 `df`；若 `/opt/venv` 或 runner 不存在，先核對模板映像，不用 pip 重裝一套推論依賴。

## 2. 建立操作目錄，傳送啟動器與 .env

**遠端 Bash**：

```bash
mkdir -p /workspace/stretto-run /data/hf /data/cache /results/logging /results/terminal /results/host-info
chmod 700 /workspace/stretto-run
df -h /data /results
command -v tmux
```

`/data` 所在檔案系統在每次 doctor/run/profile 前需至少 100GiB free。模型與 cache 用 `/data`，結果用 `/results`；同一 instance 保留這些路徑，destroy 前必須備份。這些路徑不是獨立持久卷的保證。

若缺 tmux，以 root 身分只補操作工具（若不是 root，使用主機提供的 sudo 權限）：

```bash
apt-get update
apt-get install -y --no-install-recommends tmux
```

另開 **本機 PowerShell**，傳送本機新啟動器與已填好的 `.env`。此方式不需要等待 GitHub 更新或重建映像：

```powershell
Set-Location C:\Users\yungchang\Desktop\lin_chen\SemBench-Stretto_experiment
scp -i "$env:USERPROFILE/.ssh/id_ed25519_stretto_vast" -o IdentitiesOnly=yes -P PORT scripts/reproduce_in_container.py USER@HOST:/workspace/stretto-run/reproduce_in_container.py
scp -i "$env:USERPROFILE/.ssh/id_ed25519_stretto_vast" -o IdentitiesOnly=yes -P PORT .env USER@HOST:/workspace/stretto-run/.env
```

若遠端已有 `.env`，先確認是否要更新，不要無意覆寫其他實驗憑證。不要將私鑰傳到遠端。

**遠端 Bash**：

```bash
chmod 600 /workspace/stretto-run/.env
tmux new -s stretto
```

如果提示 session 已存在，使用 `tmux attach -t stretto`。若 Vast 登入時已自動進入 tmux，可以直接使用現有 session（`echo "$TMUX"` 有值時不必再開巢狀 tmux）。按 Ctrl+B、放開、D 可離開並讓工作繼續；記下使用的 session 名稱，重連後 attach 同一個。

helper 只讀 `.env` 的 `HF_TOKEN`、`OPENAI_API_KEY`，接受 Windows BOM／CRLF 和一層包覆引號，不執行 shell 表達式。key 值後不要加行內註解。原 `.env` 的 `STRETTO_IMAGE`、相對 data/results 路徑不套用到此路線。

## 3. 先 inspect、微型 CUDA 測試，再 doctor

下列都是 **遠端 Bash**，使用絕對路徑，不依賴 VS Code 當前工作目錄。逐條執行，失敗就停在該步。

```bash
/opt/venv/bin/python /workspace/stretto-run/reproduce_in_container.py inspect
/opt/venv/bin/python -m pytest /opt/stretto/tests/test_profiling.py -k cuda_real_tiny_llama -q
/opt/venv/bin/python /workspace/stretto-run/reproduce_in_container.py doctor
```

- `inspect`：不用 key、不呼叫外部 API，檢查文字 operators、8B/3B、batch=1 與 100 篇設定。
- 微型 CUDA 測試：隨機小模型、不下載權重、不呼叫 OpenAI；應 passed。skipped 不代表 GPU 驗收成功。
- `doctor`：執行 CUDA／BF16 FlashAttention kernel，檢查容量與磁碟，並下載 HF config/tokenizer、呼叫 GPT 與 embeddings 驗證權限；會有小額 API 費用。獨立 `access` 可透過同一 helper 呼叫，但 doctor 已包含這項檢查。

在 VS Code 可開啟 `/workspace/stretto-run` 作為操作資料夾；映像程式位於 `/opt/stretto/source`，結果在 `/results`。不必重新 clone 來執行既有實驗，也不要把其他 checkout 加到 PYTHONPATH。

保存非機密環境紀錄：

```bash
uname -a > /results/host-info/uname.txt
nvidia-smi -q > /results/host-info/nvidia-smi.txt
df -h /data /results /dev/shm > /results/host-info/disk-shm.txt
cp /opt/stretto/configs/single_gpu.json /results/host-info/single_gpu.json
cp /opt/stretto/source_provenance.json /results/host-info/source_provenance.json
```

每次 helper 啟動另寫 `host-info/container-launch-*.json`，保存 launcher／主要程式與設定 SHA256。容器內沒有主機 Docker daemon 可查實際 image ID，因此不捏造 `image_id` 或 `source_revision`；run manifest 可能為 null。部署紀錄的 `expected_template_image` 是預期模板值，不是實測 digest，請另保留 Vast 的模板／instance 資訊。Vast 的 SSH 安裝流程也可能增加套件，實際 `pip-freeze.txt` 隨 run 保存。

## 4. 第一輪：100 篇 baseline

在 tmux 裡執行：

```bash
set -o pipefail
STRETTO_LOG="/results/terminal/run-$(date -u +%Y%m%dT%H%M%SZ).log"
/opt/venv/bin/python /workspace/stretto-run/reproduce_in_container.py run 2>&1 | tee "$STRETTO_LOG"
STRETTO_EXIT=${PIPESTATUS[0]}
printf 'run exit code: %s\n' "$STRETTO_EXIT"
```

runner 會再做 doctor、下載固定 revision 權重，依序測試 3B、8B、同卡共存，準備 metadata／KV，跑 1 個上游固定 benchmark query 與 8B vanilla silver reference。額外自然語言 smoke 預設關閉，只有明確加 `--nl-smoke` 才執行；manifest 記錄 `nl_smoke_enabled`。不要另外手動啟動模型 server，也不要同時執行兩輪。

每輪完成後，從**當輪 log** 取得結果路徑並檢查；不要以目錄排序猜最後一次成功結果：

```bash
STRETTO_RUN="$(sed -n 's/^Results: //p' "$STRETTO_LOG" | tr -d '\r' | tail -n 1)"
/opt/venv/bin/python - "$STRETTO_RUN" <<'PY'
import json, sys
from pathlib import Path
if not sys.argv[1]:
    raise SystemExit('這輪沒有 Results 路徑；請查看當輪終端 log。')
run = Path(sys.argv[1])
manifest = run / 'manifest.json'
if not manifest.is_file():
    raise SystemExit('這輪沒有 manifest；請查看終端 log。')
m = json.loads(manifest.read_text())
print('Run:', run)
print('Status:', m.get('status'))
print('Backend errors:', m.get('backend_errors'))
print('Wall seconds:', m.get('total_wall_s'))
print('Cache existed:', m.get('cache_existed_before_run'))
if (run / 'metrics.csv').exists():
    print((run / 'metrics.csv').read_text())
if m.get('status') != 'completed' or m.get('backend_errors') or (run / 'failure.txt').exists():
    raise SystemExit('執行尚未通過；查看 failure.txt，不擴大工作量。')
PY
```

completed 不代表 precision/recall 達到 0.9，另讀 metrics、queries、benchmark 與 silver 結果。目前評估相對於模型 silver reference，並非人工真值。

## 5. 同機詳細 profiling

前一輪完成且服務退出後執行：

```bash
nvidia-smi
set -o pipefail
STRETTO_LOG="/results/terminal/profile-$(date -u +%Y%m%dT%H%M%SZ).log"
/opt/venv/bin/python /workspace/stretto-run/reproduce_in_container.py profile 2>&1 | tee "$STRETTO_LOG"
STRETTO_EXIT=${PIPESTATUS[0]}
printf 'profile exit code: %s\n' "$STRETTO_EXIT"
```

重跑第 4 節結果檢查，再查看：

```bash
ls -lh "$STRETTO_RUN/profiling"
head -n 8 "$STRETTO_RUN/profiling/batches.csv"
/opt/venv/bin/python -m json.tool "$STRETTO_RUN/profiling/summary.json"
```

prefill/decode、搬送、RAM/VRAM、Linux faults、trace 定義沿用 [profiling.md](profiling.md)。核對 trace_coverage、limitations；cpu_only 不能當成有效 CUDA trace，not_captured 先核對路徑是否曾執行。容器看不到的主機 PID／資源要保留限制，不能解讀成整台主機沒有其他負載。

## 6. Nsight 在此 instance 的限制

目前這個一般映像未包含獨立 Nsight CLI。**不要照 VM 手冊在容器裡 docker build，也不要把 PyTorch trace 當成已量測 CUDA Unified Memory faults。** helper 保留 profile-nsys 入口，但缺少工具會失敗，現在先完成 run 與 profile。

若要在同一 instance 補 Nsight，需要另外依固定版本安裝方案準備 CLI，記錄套件版本，檢查 CUDA/NVTX/UM collector 與主機權限；或改用預先建好的 Nsight 映像建立另一個 instance。此項仍 pending，不自動加入 privileged 權限。Linux minor/major faults 與 CUDA UM faults 是不同指標。

## 7. 在同一張 GPU 重跑與完整資料

程序退出後會釋放其 RAM/VRAM；OS page cache 可能保留，不必重租或強制清 RAM。首次 run 可能含下載與 cache 建立，後續 profile 可能是 warm，不能直接以兩輪總時間差當 profiling overhead。比較 overhead 前另跑相同 cache 狀態的 warm baseline。

100 篇的品質、容量、trace 通過後，才建立完整 728 篇設定：

```bash
df -h /data /results
du -sh /data/hf /data/cache /results
/opt/venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('/workspace/stretto-run/single_gpu_full.json')
if p.exists():
    raise SystemExit('完整設定已存在，請先檢查，不覆寫。')
c = json.loads(Path('/opt/stretto/configs/single_gpu.json').read_text())
c['name'] = 'llama8b_3b_rotowire_full_v1'
c['reports_limit'] = 728
p.write_text(json.dumps(c, indent=2) + '\n')
PY
/opt/venv/bin/python /workspace/stretto-run/reproduce_in_container.py inspect --config /workspace/stretto-run/single_gpu_full.json
```

檢查 inspect 成功且顯示 728。新 config hash 會使用不同 cache namespace，可能重新建立 metadata/KV，先依 100 篇實測確認空間。完整資料仍是相同 1 個 query，並非論文完整 sweep。

```bash
set -o pipefail
STRETTO_LOG="/results/terminal/full-run-$(date -u +%Y%m%dT%H%M%SZ).log"
/opt/venv/bin/python /workspace/stretto-run/reproduce_in_container.py run --config /workspace/stretto-run/single_gpu_full.json 2>&1 | tee "$STRETTO_LOG"
STRETTO_EXIT=${PIPESTATUS[0]}
printf 'full run exit code: %s\n' "$STRETTO_EXIT"
```

重跑第 4 節結果檢查。完整 run 成功後，如需完整 profile，使用同一 --config，將 action 改成 profile，log 名稱改為 full-profile；仍須獨立驗收。

## 8. 備份，再結束租借

**遠端 Bash**，等所有 run 與 collector 完成：

```bash
cp /workspace/stretto-run/reproduce_in_container.py /results/host-info/reproduce_in_container.py
if [ -f /workspace/stretto-run/single_gpu_full.json ]; then
  cp /workspace/stretto-run/single_gpu_full.json /results/host-info/single_gpu_full.json
fi
STRETTO_ARCHIVE="/workspace/stretto-results-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
tar -czf "$STRETTO_ARCHIVE" -C / results
sha256sum "$STRETTO_ARCHIVE" > "$STRETTO_ARCHIVE.sha256"
printf '%s\n%s.sha256\n' "$STRETTO_ARCHIVE" "$STRETTO_ARCHIVE"
```

備份包含 `/results`，不含 `.env`、私鑰或 `/data` 的大型模型/cache。磁碟需有建立壓縮檔的空間。另開 **本機 PowerShell**，將 `REMOTE_ARCHIVE` 換成上述完整路徑，`ARCHIVE_NAME` 換成檔名：

```powershell
scp -i "$env:USERPROFILE/.ssh/id_ed25519_stretto_vast" -o IdentitiesOnly=yes -P PORT USER@HOST:REMOTE_ARCHIVE .
scp -i "$env:USERPROFILE/.ssh/id_ed25519_stretto_vast" -o IdentitiesOnly=yes -P PORT USER@HOST:REMOTE_ARCHIVE.sha256 .
Get-FileHash -Algorithm SHA256 .\ARCHIVE_NAME.tar.gz
Get-Content .\ARCHIVE_NAME.tar.gz.sha256
tar -tzf .\ARCHIVE_NAME.tar.gz
```

核對 hash 與壓縮內容後再結束租借。需要保留模型或 KV cache 時另外備份 `/data`。
