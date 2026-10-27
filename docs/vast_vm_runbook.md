# 租到 Vast VM 後：從 SSH 到實驗與結果備份

> **目前選定預建 image 路線，請使用 [Vast 預建 image 與一鍵批次手冊](vast_container_runbook.md)。** 不再使用 CUDA 模板現場編譯；下方 VM／其他路線僅作參考。

更新日期：2026-10-22。這是一份分段執行的操作手冊，請按順序執行；出錯先停在該步。不要把整份文件當成單一腳本一次執行。

> 若現有 instance 是 NVIDIA CUDA／vastai/base-image，它尚不包含 Stretto。請先依上方預建 image 手冊核對部署；不要在一般 CUDA 容器內執行本頁的 Docker 指令。

> **先確認租用的模板。若使用 `Stretto SSH` 自訂模板，映像為 `ghcr.io/hazcashi/sembench-stretto_experiment:436207bd211943709930577ec44895d8a1db0cd5`，請改讀 [Vast 一般容器操作手冊](vast_container_runbook.md)。** SSH／VS Code 設定可沿用本頁第 0.1、1 節；登入後跳到容器手冊，不執行下方 VM 專用的 Docker、clone、掛載或備份命令。

## 0. 適用環境與目前驗證狀態

- 使用 **Linux x86_64 VM，登入後能自行執行 Docker**。Vast 一般 Docker-based instance 需另行調整啟動方式，不適用這份 launcher 流程。
- 租前優先選官方 [Ubuntu VM 模板](https://docs.vast.ai/guides/instances/virtual-machines)，並先在 Vast 帳號加入 SSH public key。保留模板原本的 `vastai/kvm` image；登入 VM 後才拉取 Stretto image。
- 目標單張約 48GB Ampere／Ada GPU，例如 RTX A6000、RTX 6000 Ada、L40S；建議分配 RAM 至少 64GB、約 8 vCPU。要留在同機擴大資料，可先抓約 300GB 磁碟，再依實測 cache 大小評估。
- **每次 doctor／run／profile 前，data 所在檔案系統都必須仍有至少 100GiB 可用空間**。這不是磁碟總容量 100GB。
- 本手冊使用已發佈的新版一般映像，來源程式 `436207bd211943709930577ec44895d8a1db0cd5`；本機 Docker 測試 **36 passed、1 skipped**，無 GPU `inspect` 與匿名 registry manifest 驗證通過。[發佈流程紀錄](https://github.com/hazcashi/SemBench-Stretto_experiment/actions/runs/39447527105)
- 真正的 GPU／HF／OpenAI 存取、雙模型、100 篇實驗、Nsight、完整資料與 overhead 仍待現場驗收。憑證填妥不等於已通過存取測試。

除標記為 **本機 PowerShell** 的區塊外，下方命令都在 **租借 VM 的 Bash** 執行。若重開 shell，需要重新設定文件中的 `STRETTO_IMAGE` 等 shell 變數；`.env` 與磁碟上的資料會保留。

## 0.1 借用 Windows 電腦：先建立專用 SSH key

Vast 帳號存的是公鑰；連線電腦需要對應私鑰。原電腦無法使用時，可在借用電腦產生新的一組，再把新公鑰加入同一個 Vast 帳號，不必刪除舊公鑰。

**本機 PowerShell**（VS Code 的本機終端也可以）。使用獨立檔名，保留借用電腦原有的金鑰：

```powershell
$strettoKeyPath = Join-Path $env:USERPROFILE '.ssh/id_ed25519_stretto_vast'
New-Item -ItemType Directory -Force (Split-Path $strettoKeyPath) | Out-Null
if ((Test-Path -LiteralPath $strettoKeyPath) -or (Test-Path -LiteralPath ($strettoKeyPath + '.pub'))) {
  throw '專用 key 已存在，請先確認用途，不要覆寫。'
}
ssh-keygen -t ed25519 -C 'stretto-vast-borrowed-pc' -f $strettoKeyPath
```

提示 `Enter passphrase` 時，由你在終端輸入自行設定的保護密碼，再輸入一次。輸入時不顯示字元是正常現象；這是金鑰保護密碼，與 Vast 帳號密碼不同。

- `id_ed25519_stretto_vast`：私鑰，留在這台電腦，不上傳 GitHub、Vast、GPU VM 或聊天。
- `id_ed25519_stretto_vast.pub`：公鑰，下一步貼到 Vast。

複製公鑰至剪貼簿：

```powershell
Get-Content "$env:USERPROFILE/.ssh/id_ed25519_stretto_vast.pub" | Set-Clipboard
```

開啟 [Vast Account](https://cloud.vast.ai/account/)，在 SSH Keys 區域新增公鑰，貼上整行 `ssh-ed25519 ... stretto-vast-borrowed-pc` 並儲存。保留原電腦的公鑰。

**先加入公鑰，再建立 Ubuntu VM。** 新增帳號公鑰不會自動更新既有 instance；Vast 官方說明 VM 的平台換鑰流程需要重建 VM，因此租前完成。[SSH 官方說明](https://docs.vast.ai/guides/instances/connect/ssh)、[VM 官方說明](https://docs.vast.ai/guides/instances/virtual-machines)

## 1. 從本機 SSH 登入與 VS Code Remote SSH

**本機 PowerShell**，將 `PORT`、`USER`、`HOST` 換成 Vast 提供的 SSH 連線資訊：

```powershell
ssh -i "$env:USERPROFILE/.ssh/id_ed25519_stretto_vast" -o IdentitiesOnly=yes -p PORT USER@HOST
```

`USER`、`HOST`、`PORT` 都以該 VM 的 Vast SSH 連線資訊為準，不自行假設帳號或 port。若有 Direct SSH 可優先使用。首次連線確認主機資訊與可取得的 host fingerprint 後接受提示，再輸入剛才的 key passphrase。

確認上述 PowerShell 連線成功，再設定 VS Code。若尚未安裝 Microsoft Remote - SSH，可在擴充套件搜尋 `ms-vscode-remote.remote-ssh` 安裝。

在**本機 VS Code 視窗**按 `Ctrl+Shift+P` → `Remote-SSH: Open SSH Configuration File...` → 選使用者的 `C:\Users\<你的使用者>\.ssh\config`，在檔案末尾新增以下區塊，保留原有內容，並替換三個佔位值：

```sshconfig
Host stretto-vast
    HostName HOST
    User USER
    Port PORT
    IdentityFile ~/.ssh/id_ed25519_stretto_vast
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

按 `Ctrl+Shift+P` → `Remote-SSH: Connect to Host...` → `stretto-vast`；平台選 Linux，依提示輸入 key passphrase，等待 VS Code Server 安裝。左下角應顯示 `SSH: stretto-vast`。往後本機也能用 `ssh stretto-vast` 連線。[VS Code 官方說明](https://code.visualstudio.com/docs/remote/ssh)

連線後，VS Code 的終端就是遠端 Bash。**一般 Stretto SSH 容器現在跳到 [容器手冊](vast_container_runbook.md) 第 1 步。** 只有 VM 才繼續下面的環境檢查和第 2 步 clone。Remote SSH 不會自動同步本機專案或 `.env`；依所選路線的手冊另傳。

歸還借用電腦前，結束實驗並確認不再需要這組 key，再從 Vast 移除這組專用公鑰，從本機 config 移除 `stretto-vast` 區塊，刪除這組專用私鑰與 `.pub`；保留原有金鑰。實驗憑證與備份另行妥善處理。

登入後，在 **VM Bash** 檢查：

```bash
uname -m
nvidia-smi
docker version
docker info
git --version
python3 --version
tmux -V
```

預期 `uname -m` 為 `x86_64`；`docker version` 有 Client 與 Server，`docker info` 能回覆。若 Docker 不存在、無法連線 daemon 或 permission denied，先釐清 VM／daemon／使用者權限，不進入後續步驟，也不把一般 GPU 容器誤當成完整 VM。

如果只缺 Git、tmux、nano、Python，Ubuntu VM 可安裝這些操作工具；推論依賴已在 image 裡：

```bash
if [ "$(id -u)" -eq 0 ]; then
  apt-get update
  apt-get install -y git tmux nano python3
else
  sudo apt-get update
  sudo apt-get install -y git tmux nano python3
fi
```

## 2. 開啟 tmux，下載專案

```bash
tmux new -s stretto
```

進入 tmux 後：

```bash
mkdir -p "$HOME/stretto-work"
cd "$HOME/stretto-work"
git clone --recurse-submodules https://github.com/hazcashi/SemBench-Stretto_experiment.git
cd SemBench-Stretto_experiment
git rev-parse HEAD
git submodule status
pwd
```

記下 `pwd` 的絕對路徑，後面傳 `.env` 和取回結果會用到。若已有 checkout，先檢查 `git status`，確認自己的修改，再使用 `git pull --ff-only`、`git submodule update --init --recursive`；不要重新 clone 到同名目錄或用 reset／clean 覆蓋資料。

離開 tmux 但保持實驗執行：按 **Ctrl+B，放開，再按 D**。SSH 重連後使用：

```bash
tmux attach -t stretto
```

## 3. 直接拉取已發佈的新版 image

這一步不需要重新編譯 FlashAttention，也不需要 HF／OpenAI key。

```bash
STRETTO_IMAGE='ghcr.io/hazcashi/sembench-stretto_experiment@sha256:afe378f49f356403d2769c6d49e81cb9beb859773817765125da16b6841bbb68'
docker pull "$STRETTO_IMAGE"
docker image inspect "$STRETTO_IMAGE" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
docker run --rm --gpus device=0 --entrypoint nvidia-smi "$STRETTO_IMAGE"
```

來源標籤應為 `436207bd211943709930577ec44895d8a1db0cd5`。容器應看到選定的 GPU；若主機有多卡，這份指令只暴露 GPU 0。這一步只證明容器可取得 GPU，真正的 CUDA／FlashAttention kernel 後面由 doctor 驗證。

**不要以 `docker/image.txt` 取代上面的 digest**：它仍保留舊 baseline，沒有這次詳細 profiling。文件 commit 可能比 image 的程式 commit 新，不能只因兩者不同就認為 image 有誤。

建立資料與紀錄目錄，並確認剩餘空間：

```bash
mkdir -p data/hf data/cache results/host-info results/terminal
df -h .
docker info --format '{{.DockerRootDir}}'
git rev-parse HEAD > results/host-info/checkout_commit.txt
git submodule status > results/host-info/submodules.txt
uname -a > results/host-info/uname.txt
nvidia-smi -q > results/host-info/nvidia-smi.txt
docker info > results/host-info/docker-info.txt
docker image inspect "$STRETTO_IMAGE" --format '{{json .Config.Labels}}' > results/host-info/image-labels.json
cp configs/single_gpu.json results/host-info/single_gpu.json
```

若 Docker 儲存與 `data/` 在不同磁碟，也要檢查 Docker 所在磁碟的可用空間。本手冊後續固定使用專案根目錄的 `data/`、`results/`。

## 4. 將憑證放入租借主機的 .env

本機 `.env` 不會隨 GitHub clone 出現。以下兩種方式擇一；不要把憑證貼到聊天或終端命令參數。

### 方式 A：傳送本機已填好的 .env

另開 **本機 PowerShell**，進入本機專案。將 SSH 資訊和 `REMOTE_PROJECT` 換成剛剛 VM 上 `pwd` 顯示的專案絕對路徑：

```powershell
Set-Location C:\Users\yungchang\Desktop\lin_chen\SemBench-Stretto_experiment
scp -i "$env:USERPROFILE/.ssh/id_ed25519_stretto_vast" -o IdentitiesOnly=yes -P PORT .env USER@HOST:REMOTE_PROJECT/.env
```

### 方式 B：在 VM 重新填寫

在 **VM Bash**、專案根目錄執行；已有 `.env` 時保留原內容：

```bash
if [ ! -f .env ]; then
  (umask 077; cp .env.example .env)
fi
nano .env
```

填好 `HF_TOKEN` 和 `OPENAI_API_KEY`。不要執行 `cat .env`、`printenv` 或開啟 `set -x` 來檢查憑證。

### 統一非機密欄位，保留兩個 key

回到 **VM Bash**。這段只更新 image／資料位置、移除 Windows 換行與 BOM，不顯示憑證；若本來自訂了資料路徑，請先決定是否採用本手冊的固定路徑：

```bash
chmod 600 .env
python3 - <<'PY'
from pathlib import Path
p = Path('.env')
updates = {
    'STRETTO_IMAGE': 'ghcr.io/hazcashi/sembench-stretto_experiment@sha256:afe378f49f356403d2769c6d49e81cb9beb859773817765125da16b6841bbb68',
    'STRETTO_NSYS_IMAGE': 'stretto-nsys:local',
    'STRETTO_DATA_DIR': './data',
    'STRETTO_RESULTS_DIR': './results',
}
lines = [line for line in p.read_text(encoding='utf-8-sig').splitlines()
         if line.split('=', 1)[0] not in updates]
lines.extend(f'{key}={value}' for key, value in updates.items())
p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
values = dict(line.split('=', 1) for line in lines
              if '=' in line and not line.lstrip().startswith('#'))
missing = [key for key in ('HF_TOKEN', 'OPENAI_API_KEY') if not values.get(key, '').strip()]
if missing:
    raise SystemExit('請先填妥：' + ', '.join(missing))
print('憑證欄位已填入；尚待 access 驗證。非機密設定已更新。')
PY
```

`stretto-nsys:local` 是選配步驟才會建立的 tag，先填入不會影響一般 run。若本機曾 export 過其他 image／資料位置，先讓 launcher 使用剛填好的 `.env`：

```bash
unset STRETTO_IMAGE STRETTO_NSYS_IMAGE STRETTO_DATA_DIR STRETTO_RESULTS_DIR
```

## 5. inspect、存取檢查、GPU kernel 檢查

依序執行，每一條成功後才往下：

```bash
bash scripts/reproduce.sh inspect
bash scripts/reproduce.sh access
bash scripts/reproduce.sh doctor
```

- `inspect`：不呼叫 OpenAI，不下載權重；核對 3B／8B、batch=1、100 篇與文字 operators。
- `access`：下載兩模型的 config／tokenizer，實際呼叫兩個 GPT snapshot 和 embeddings；有小額 API 費用，成功應為 `status: passed`。
- `doctor`：確認單張可見 GPU、容量、100GiB free，實際執行 CUDA tensor／BF16 FlashAttention，再做存取檢查。它也會產生小額 API 費用；後續 run 仍會重做 doctor。

遇到 HF 401／403、OpenAI 401／403／429，保留 `results/<run-id>/failure.txt` 和錯誤，先修存取／額度／速率問題，不更換模型冒充相同實驗。

再執行不下載權重、不呼叫 OpenAI 的微型隨機 Llama CUDA 測試：

```bash
STRETTO_IMAGE='ghcr.io/hazcashi/sembench-stretto_experiment@sha256:afe378f49f356403d2769c6d49e81cb9beb859773817765125da16b6841bbb68'
docker run --rm --gpus device=0 --entrypoint python "$STRETTO_IMAGE" \
  -m pytest /opt/stretto/tests/test_profiling.py -k cuda_real_tiny_llama -q
```

這項在 GPU 主機應為 passed。若 skipped 或 failed，先診斷原因，不視為 CUDA 驗收成功。

## 6. 第一輪：100 篇 baseline

不需要手動啟動 3B／8B／embedding servers。Runner 會管理程序、先各別驗證兩模型，再測同卡共存，準備 metadata／KV，執行自然語言 smoke、固定查詢與 8B vanilla silver reference。

```bash
set -o pipefail
STRETTO_LOG="results/terminal/run-$(date -u +%Y%m%dT%H%M%SZ).log"
bash scripts/reproduce.sh run 2>&1 | tee "$STRETTO_LOG"
STRETTO_EXIT=${PIPESTATUS[0]}
printf 'run exit code: %s\n' "$STRETTO_EXIT"
```

這會在需要時下載固定 revision 的模型權重。首次執行的總時間包含前置成本；非零 exit code 時先停下，保存 log。

以下從**剛剛這份 log** 取得 run ID，避免誤讀上一次成功結果。每輪 run／profile／profile-nsys 後都可以重跑此檢查；變數 `STRETTO_LOG` 要指向該輪 log：

```bash
STRETTO_RUN_ID="$(sed -n 's#^Results: /results/##p' "$STRETTO_LOG" | tr -d '\r' | tail -n 1)"
STRETTO_RUN="results/$STRETTO_RUN_ID"
python3 - "$STRETTO_RUN" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
manifest = run / 'manifest.json'
if not manifest.is_file():
    raise SystemExit('找不到這輪 manifest；請檢查終端 log，不要改讀舊 run。')
m = json.loads(manifest.read_text())
print('Run:', run)
print('Status:', m.get('status'))
print('Backend errors:', m.get('backend_errors'))
print('Total wall seconds:', m.get('total_wall_s'))
print('Cache existed:', m.get('cache_existed_before_run'))
if (run / 'failure.txt').exists():
    print('請查看：', run / 'failure.txt')
metrics = run / 'metrics.csv'
if metrics.exists():
    print(metrics.read_text())
if m.get('status') != 'completed' or m.get('backend_errors') or (run / 'failure.txt').exists():
    raise SystemExit('這輪尚未通過執行檢查，先不要擴大工作量。')
PY
```

`completed` 不等於 precision／recall 已達 0.9；另外閱讀 `metrics.csv`、`queries.json`、`benchmark_0.csv` 和 `silver_0.csv`。目前分數是相對 8B silver reference，並非人工真值。

## 7. 同一台機器跑詳細 profiling

先確認上一輪完成、舊服務已退出，再執行；仍是同樣的 100 篇，不會自動擴大資料：

```bash
nvidia-smi
STRETTO_LOG="results/terminal/profile-$(date -u +%Y%m%dT%H%M%SZ).log"
bash scripts/reproduce.sh profile 2>&1 | tee "$STRETTO_LOG"
STRETTO_EXIT=${PIPESTATUS[0]}
printf 'profile exit code: %s\n' "$STRETTO_EXIT"
```

重跑第 6 步的結果檢查，讓 `STRETTO_RUN` 指向這輪結果。再檢查詳細輸出：

```bash
ls -lh "$STRETTO_RUN/profiling"
head -n 8 "$STRETTO_RUN/profiling/batches.csv"
python3 - "$STRETTO_RUN/profiling/summary.json" <<'PY'
import json, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text())
print('Trace coverage:')
print(json.dumps(s.get('trace_coverage'), indent=2))
print('Limitations:')
print(json.dumps(s.get('limitations'), indent=2))
PY
```

確認代表 trace 有 CUDA 活動。`cpu_only` 不是有效 CUDA trace；`not_captured` 需要核對該模型／路徑是否真的執行。最多擷取兩模型各 vanilla/direct、KV 的首個 benchmark batch；可能是 optimizer 試跑，沒有走到的路徑不補造 trace。

## 8. 選配：Nsight Systems／CUDA Unified Memory faults

**這個獨立映像與 GPU collector 仍待現場驗收。** 完成一般 profile 後才做，與 PyTorch Profiler 分開執行，不自動加入 privileged 權限。

從同一個已驗證新版一般映像衍生，建置時不重編譯推論依賴：

```bash
STRETTO_IMAGE='ghcr.io/hazcashi/sembench-stretto_experiment@sha256:afe378f49f356403d2769c6d49e81cb9beb859773817765125da16b6841bbb68'
docker build --platform linux/amd64 -f docker/Dockerfile.profiling \
  --build-arg BASE_IMAGE="$STRETTO_IMAGE" -t stretto-nsys:local .
docker run --rm --entrypoint nsys stretto-nsys:local --version
docker run --rm --gpus device=0 --entrypoint nsys stretto-nsys:local status --environment
```

檢查診斷內容；CPU sampling 等未支援不一定代表 CUDA trace 不可用。需要確認的是本輪所需的 CUDA／NVTX／UM collector，無法判斷時先保存輸出。

```bash
STRETTO_LOG="results/terminal/profile-nsys-$(date -u +%Y%m%dT%H%M%SZ).log"
bash scripts/reproduce.sh profile-nsys 2>&1 | tee "$STRETTO_LOG"
STRETTO_EXIT=${PIPESTATUS[0]}
printf 'profile-nsys exit code: %s\n' "$STRETTO_EXIT"
```

重跑第 6 步檢查後，閱讀這輪的：

```bash
ls -lh "$STRETTO_RUN/profiling/nsight"
python3 -m json.tool "$STRETTO_RUN/profiling/nsight_environment.json"
python3 -m json.tool "$STRETTO_RUN/profiling/cuda_page_faults.json"
```

等命令完成並返回 prompt，讓 Nsight 有時間匯出 `.nsys-rep`／SQLite，再進入下一輪。`range_completed` 不等於報告已驗收；需要有效檔案、CUDA 資料與 collector 狀態。沒有 UM 表或功能不支援，不能當成零 faults。

## 9. 重跑、RAM／VRAM、cold／warm 與 overhead

- 可以在同一個租借 instance 反覆執行不同命令。每次產生新的 run ID，結果 cache 停用，HF 權重與應用 cache 保存在磁碟。
- 程序退出後會釋放其 RAM／VRAM；先用 `nvidia-smi` 核對。OS page cache 可能保留，不必為了重跑重開機。本指南不執行系統級 drop_caches。
- 第一次 run 可能含下載與建 KV；後面的 profile 可能是 warm。不要直接用這兩次總時間宣稱 profiling overhead。
- 若要比較穩定重用下的 overhead，先完成首輪，之後再用第 6 步命令跑一次 warm baseline，再依序做 profile／profile-nsys。保持相同 image、設定、資料、cache 目錄，另外記錄各輪實際 cache 狀態；這仍不保證 OS cache 完全一致。
- 看 `queries.json` 的階段時間與結果，再比較詳細量測；不要把重疊 spans 直接相加當 wall time。
- 若要中止，先按 Ctrl+C 並等待清理／報告匯出；保留失敗紀錄，不用 `kill -9` 或刪除 data 當作一般重跑步驟。

## 10. 選配：100 篇通過後，在同機擴到完整 728 篇

**以下使用既有 runner 的 `--config` 入口與額外設定檔掛載，尚未完成 GPU 驗收。** 100 篇結果、品質與容量確認後才執行。完整資料仍只有相同的 1 個固定 query，不是論文完整 query／參數 sweep。

先查看空間；依首輪量測估算新增 KV／trace，而非只看權重是否放得下：

```bash
df -h .
du -sh data/hf data/cache results
```

建立另一份設定，保留原本 100 篇配置；已有同名檔案時先檢查，不覆寫：

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('configs/single_gpu_full.json')
if p.exists():
    raise SystemExit('設定檔已存在，請先檢查 configs/single_gpu_full.json。')
config = json.loads(Path('configs/single_gpu.json').read_text())
config['name'] = 'llama8b_3b_rotowire_full_v1'
config['reports_limit'] = 728
p.write_text(json.dumps(config, indent=2) + '\n')
print('已建立', p, '；optimizer sample_size 仍為', config['sample_size'])
PY
```

新的設定 hash 會使用不同 cache namespace，因此可能重新建立 metadata／KV；HF 權重仍共用 `data/hf/`。

在目前 Bash 定義下列 helper。它沿用 launcher 的 GPU／環境／掛載與 image metadata，額外掛載完整資料設定：

```bash
STRETTO_IMAGE='ghcr.io/hazcashi/sembench-stretto_experiment@sha256:afe378f49f356403d2769c6d49e81cb9beb859773817765125da16b6841bbb68'
stretto_full() {
  local action="${1:?請指定 inspect、run、profile 或 profile-nsys}"
  local image="$STRETTO_IMAGE"
  case "$action" in
    inspect|run|profile) ;;
    profile-nsys) image='stretto-nsys:local' ;;
    *) printf '不支援的 action: %s\n' "$action"; return 2 ;;
  esac
  docker run --rm --init --gpus device=0 --shm-size 2g --env-file .env \
    -e STRETTO_IMAGE_REF="$image" \
    -e STRETTO_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$image")" \
    -e STRETTO_SOURCE_REVISION="$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$image")" \
    --mount "type=bind,source=$PWD/data,target=/data" \
    --mount "type=bind,source=$PWD/results,target=/results" \
    --mount "type=bind,source=$PWD/configs/single_gpu_full.json,target=/opt/stretto/configs/single_gpu_full.json,readonly" \
    "$image" "$action" --config /opt/stretto/configs/single_gpu_full.json
}
stretto_full inspect
```

確認顯示 `reports_limit: 728`，再跑完整資料 baseline：

```bash
STRETTO_LOG="results/terminal/full-run-$(date -u +%Y%m%dT%H%M%SZ).log"
stretto_full run 2>&1 | tee "$STRETTO_LOG"
STRETTO_EXIT=${PIPESTATUS[0]}
printf 'full run exit code: %s\n' "$STRETTO_EXIT"
```

重跑第 6 步的結果檢查。成功且空間足夠後，再做完整資料 profile：

```bash
STRETTO_LOG="results/terminal/full-profile-$(date -u +%Y%m%dT%H%M%SZ).log"
stretto_full profile 2>&1 | tee "$STRETTO_LOG"
STRETTO_EXIT=${PIPESTATUS[0]}
printf 'full profile exit code: %s\n' "$STRETTO_EXIT"
```

如需完整資料 Nsight，在第 8 步驗收通過後執行；完成後重跑第 6 步檢查及第 8 步報告檢查：

```bash
STRETTO_LOG="results/terminal/full-profile-nsys-$(date -u +%Y%m%dT%H%M%SZ).log"
stretto_full profile-nsys 2>&1 | tee "$STRETTO_LOG"
STRETTO_EXIT=${PIPESTATUS[0]}
printf 'full profile-nsys exit code: %s\n' "$STRETTO_EXIT"
```

這些都使用同一個 instance，不必重新租 GPU。

## 11. 在結束租借前備份

確認所有程序與 Nsight 匯出完成。**VM Bash**、專案根目錄：

```bash
git rev-parse HEAD > results/host-info/checkout_commit_at_backup.txt
if [ -f configs/single_gpu_full.json ]; then
  cp configs/single_gpu_full.json results/host-info/single_gpu_full.json
fi
STRETTO_ARCHIVE="$HOME/stretto-results-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
tar -czf "$STRETTO_ARCHIVE" results
sha256sum "$STRETTO_ARCHIVE" > "$STRETTO_ARCHIVE.sha256"
printf '下載這兩個檔案：\n%s\n%s.sha256\n' "$STRETTO_ARCHIVE" "$STRETTO_ARCHIVE"
```

這個壓縮檔包含成功／失敗 run、終端 log、upstream logging、trace 和設定紀錄；不包含 `.env` 或大型 `data/`。如需保留 HF／KV cache，另安排 persistent storage／備份，不能只下載 results 就假設 cache 也保存了。

**本機 PowerShell**，用剛剛印出的完整路徑取代 `REMOTE_ARCHIVE`，檔名取代 `ARCHIVE_NAME`：

```powershell
scp -i "$env:USERPROFILE/.ssh/id_ed25519_stretto_vast" -o IdentitiesOnly=yes -P PORT USER@HOST:REMOTE_ARCHIVE .
scp -i "$env:USERPROFILE/.ssh/id_ed25519_stretto_vast" -o IdentitiesOnly=yes -P PORT USER@HOST:REMOTE_ARCHIVE.sha256 .
Get-FileHash -Algorithm SHA256 .\ARCHIVE_NAME.tar.gz
Get-Content .\ARCHIVE_NAME.tar.gz.sha256
tar -tzf .\ARCHIVE_NAME.tar.gz
```

確認下載成功、hash 一致、壓縮檔可列出內容後，再到 Vast 結束租借。不要為了兩輪之間清 RAM 而 destroy instance。

欄位與量測限制見 [profiling.md](profiling.md)；帳號準備見 [pre_rental_checklist.md](pre_rental_checklist.md)。
