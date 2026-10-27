# Vast NVIDIA CUDA 模板：安裝、100 篇實驗、profiling 與備份

> **此頁現場安裝路線已退出目前操作流程。使用者選擇直接使用預建 image，請改讀 [預建 image 與一鍵 100＋728 篇流程](vast_container_runbook.md)。不要執行本頁 bootstrap 或重新編譯。** 以下僅保留為手動安裝參考。

更新：2026-10-22。以下是保留的手動安裝參考，適用一般 Vast SSH 容器，映像為 `vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu24.04-py312-2026-10-21`。目前選定的預建 image 路線不使用以下安裝步驟。

## 0. 先核對租到的資源

- x86_64、單張 RTX A6000／RTX 6000 Ada／L40S 等 Ampere/Ada GPU，約 48GB VRAM。此安裝的 FlashAttention build 不適用 H100／Blackwell。
- instance 實際分配 CPU RAM 建議至少 64GB；本次截圖的 86GB／96.5GB 合適，但實際雙模型峰值仍需量測。
- **建立 instance 前，Disk Space 手動設約 300GB**。優惠卡的 SSD 可用空間不等於租到的容量；建立後 Disk 應顯示約 `0 / 300GB`。32GB 不夠；已建立的 container disk 不能調大。[Vast 官方儲存說明](https://docs.vast.ai/guides/instances/storage/types)
- 300GB 是起始規劃，不保證完整 728 篇及多輪 trace 全部放得下。每次 doctor/run/profile 還要求 data 檔案系統至少 100GiB free。容器 `df` 可能顯示主機容量，所以必須同時看 Vast 配額。
- Launch mode 使用 **Interactive shell server, SSH**；保留官方模板其他設定。不要自行安裝 NVIDIA driver 或在容器內裝 Docker daemon。

**驗證界線：** 固定依賴來自已建成、CPU 測試通過的 Stretto Dockerfile。新 CUDA 模板的安裝腳本尚未完成一台全新遠端主機上的完整驗收；它會在遠端執行依賴檢查、inspect 與測試。實際 GPU、HF/OpenAI 存取、100 篇 benchmark、trace、Nsight 均需照下列步驟驗收，不能把本機語法測試當成實驗成功。

本機已驗證：三份 Vast 手冊共 53 個 Bash 區塊、12 個 PowerShell 區塊與 9 段內嵌 Python 語法；安裝腳本 Bash 語法與 32GB 配額拒絕；啟動器 6 個 CPU 測試；啟動器在既有 Stretto 映像內 `inspect` 成功列出 100 篇／3B+8B／27 operators。上述檢查不含全新 CUDA 基礎映像的依賴下載與編譯。

## 1. 本機 SSH 與上傳安裝工具

以下是 **Windows 本機 PowerShell**。將 `PORT`、`USER`、`HOST` 換成新 instance Connect 顯示的內容，**不要沿用已刪除機器的 IP/port**。專用 SSH key 設定見 [VM 手冊第 0.1、1 節](vast_vm_runbook.md)；VS Code Remote SSH 也要更新 HostName/Port。

```powershell
Set-Location C:\Users\yungchang\Desktop\lin_chen\SemBench-Stretto_experiment
ssh -i "$env:USERPROFILE/.ssh/id_ed25519_stretto_vast" -o IdentitiesOnly=yes -p PORT USER@HOST "mkdir -p /workspace/stretto-run && chmod 700 /workspace/stretto-run"
scp -i "$env:USERPROFILE/.ssh/id_ed25519_stretto_vast" -o IdentitiesOnly=yes -P PORT .\scripts\bootstrap_vast_cuda.sh .\scripts\reproduce_in_container.py USER@HOST:/workspace/stretto-run/
ssh -i "$env:USERPROFILE/.ssh/id_ed25519_stretto_vast" -o IdentitiesOnly=yes -p PORT USER@HOST
```

這兩個安裝／啟動工具直接從本機上傳，因此不依賴尚未推送的新檔案。安裝腳本會從公開 GitHub 自動抓取固定 runtime commit `436207bd211943709930577ec44895d8a1db0cd5`，以及該版 submodule；不需上傳整個專案或模型。

## 2. 遠端環境檢查與安裝

以下開始都是 **遠端 Bash**，直到明確標示本機 PowerShell。

```bash
uname -m
cat /etc/os-release
nvidia-smi
nvcc --version
python3.12 --version
free -h
df -h / /workspace /dev/shm
```

應為 x86_64、Ubuntu 24.04、Python 3.12、nvcc release 12.8，並看到租到的 GPU。`nvidia-smi` 右上角 CUDA Version 是 driver 支援上限，不要求它剛好等於 12.8。`free` 可能顯示整台主機 RAM，需對照 instance 配額及後續 resource sampler 的 cgroup 限制。

```bash
apt-get update
apt-get install -y --no-install-recommends tmux
tmux new -s stretto
```

進入 tmux 後執行。`300` 必須是 Vast 卡片顯示的**實際租用 Disk 配額 GB**，不是為了通過檢查而填的數字。腳本要求配額至少 250GB、安裝前檔案系統可用至少 150GiB；仍建議租 300GB。

```bash
cd /workspace/stretto-run
sed -i 's/\r$//' bootstrap_vast_cuda.sh
set -o pipefail
STRETTO_SETUP_LOG="/workspace/stretto-run/setup-$(date -u +%Y%m%dT%H%M%SZ).log"
bash bootstrap_vast_cuda.sh 300 2>&1 | tee "$STRETTO_SETUP_LOG"
STRETTO_SETUP_EXIT=${PIPESTATUS[0]}
printf 'setup exit code: %s\n' "$STRETTO_SETUP_EXIT"
```

**只有 exit code 0 且出現 `Installation checks passed` 才繼續。** 安裝包含：

1. apt 安裝 Python venv、C/C++ 編譯工具、Git 等；不更動主機 driver。
2. `/opt/venv` 建立獨立 Python 3.12 環境，pip 25.2，以 SHA256 lock 安裝依賴，包括 PyTorch `2.9.1+cu128`、Transformers `5.2.0`。
3. 編譯 FlashAttention `2.8.3.post2`，與 Dockerfile 相同：sm80、停用 backward、單一編譯工作；保留原推論流程。**這不是拉取已建好 Stretto image，編譯可能很久、可能以小時計；目前沒有這台機器的時間實測。** 不要因為久就另開第二份安裝。
4. 安裝固定版 kvpress／Stretto、下載與 hash 驗證 5 個 Rotowire CSV、安裝 DuckDB VSS extension。
5. 執行 pip check、inspect 與測試；有 CUDA 時包含不需下載權重的 tiny random Llama 測試。

安裝中可按 `Ctrl+B` 再 `D` 離開 tmux，工作會繼續；SSH 回來後 `tmux attach -t stretto`。安裝失敗保留 log，先處理錯誤；相同腳本支援同版重試，成功編譯的 FlashAttention 會保留。若已有來源不明的 `/opt/venv`，腳本會拒絕覆寫。不要手動刪除所有環境再盲目重裝。

```bash
cat /opt/stretto/installation.json
cp "$STRETTO_SETUP_LOG" /results/terminal/
/opt/venv/bin/python /workspace/stretto-run/reproduce_in_container.py inspect
```

應看到 100 reports、batch size 1、Llama 3B/8B，以及 27 個文字 operators；`inspect` 不下載模型、不呼叫 API。安裝紀錄位於 `/results/host-info`，含固定 source revision、依賴 hash、pip freeze、CUDA compiler 版本；模板名稱是宣告值，不冒充主機驗證過的 image digest。

## 3. 上傳兩個憑證，再做 doctor

另開 **Windows 本機 PowerShell**；本機專案 `.env` 已填 `HF_TOKEN` 與 `OPENAI_API_KEY`。不把 token 貼到聊天、不提交 Git，也不將 `.env` 印到終端。

```powershell
Set-Location C:\Users\yungchang\Desktop\lin_chen\SemBench-Stretto_experiment
scp -i "$env:USERPROFILE/.ssh/id_ed25519_stretto_vast" -o IdentitiesOnly=yes -P PORT .\.env USER@HOST:/workspace/stretto-run/.env
```

回到 **遠端 Bash / tmux**：

```bash
chmod 600 /workspace/stretto-run/.env
set -o pipefail
STRETTO_LOG="/results/terminal/doctor-$(date -u +%Y%m%dT%H%M%SZ).log"
/opt/venv/bin/python /workspace/stretto-run/reproduce_in_container.py doctor 2>&1 | tee "$STRETTO_LOG"
STRETTO_EXIT=${PIPESTATUS[0]}
printf 'doctor exit code: %s\n' "$STRETTO_EXIT"
```

doctor 檢查單 GPU、至少 44GiB VRAM、CUDA/FlashAttention 運算、磁碟、兩個 gated Llama 固定 revision、OpenAI 模型與 embedding 存取。它有少量付費 API 請求；不等同完整雙模型壓力測試。退出碼 0、當輪 `doctor.json` 正常才繼續；失敗檢查當輪 `failure.txt`。

模型不用在本機抓。第一輪 run 會自動下載固定 revision 到 `/data/hf`。使用 `meta-llama/Llama-3.1-8B-Instruct` 與 `meta-llama/Llama-3.2-3B-Instruct`，HF token 所屬帳號須已獲兩者權限。OpenAI 用於現有規劃／metadata 等路徑，不能因本地推論而省略。

## 4. 第一輪：100 篇 run

```bash
set -o pipefail
STRETTO_LOG="/results/terminal/run-$(date -u +%Y%m%dT%H%M%SZ).log"
/opt/venv/bin/python /workspace/stretto-run/reproduce_in_container.py run 2>&1 | tee "$STRETTO_LOG"
STRETTO_EXIT=${PIPESTATUS[0]}
printf 'run exit code: %s\n' "$STRETTO_EXIT"
```

固定 100 篇 Rotowire，3B/8B 個別與並行 smoke、建立必要 cache、正式固定查詢、8B silver reference。不是 100 個獨立問答，也不是論文所有 benchmark。`run` 保留低開銷資源與 Linux 缺頁紀錄；服務、結果與失敗資訊都寫入獨立 run directory。

**每一輪 run/profile/full 結束後都執行以下檢查**，使用該輪的 `STRETTO_LOG`，不要從目錄排序猜哪次成功：

```bash
STRETTO_RUN="$(sed -n 's/^Results: //p' "$STRETTO_LOG" | tr -d '\r' | tail -n 1)"
/opt/venv/bin/python - "$STRETTO_RUN" <<'PY'
import json, sys
from pathlib import Path
if not sys.argv[1]:
    raise SystemExit('No Results path; inspect the current terminal log')
run = Path(sys.argv[1])
m = json.loads((run / 'manifest.json').read_text())
print('Run:', run, 'Status:', m.get('status'))
print('Backend errors:', m.get('backend_errors'))
print('Wall seconds:', m.get('total_wall_s'))
print('Cache existed:', m.get('cache_existed_before_run'))
if (run / 'metrics.csv').exists():
    print((run / 'metrics.csv').read_text())
if m.get('status') != 'completed' or m.get('backend_errors') or (run / 'failure.txt').exists():
    raise SystemExit('Run failed: inspect failure.txt and services; do not expand workload')
PY
```

completed 只表示流程完成，不保證 precision/recall 達標；讀 metrics、queries、benchmark 與 silver 結果。品質對照是模型 silver reference，不是人工真值。

## 5. 同一台接著跑 profile

等前輪結束、服務退出再執行，不要兩輪並行。不需重租或強制清 RAM。程序退出會釋放其記憶體，但 OS page cache／磁碟 KV cache 可能保留。

```bash
nvidia-smi
df -h /data /results
du -sh /data/hf /data/cache /results
set -o pipefail
STRETTO_LOG="/results/terminal/profile-$(date -u +%Y%m%dT%H%M%SZ).log"
/opt/venv/bin/python /workspace/stretto-run/reproduce_in_container.py profile 2>&1 | tee "$STRETTO_LOG"
STRETTO_EXIT=${PIPESTATUS[0]}
printf 'profile exit code: %s\n' "$STRETTO_EXIT"
```

重跑第 4 節結果檢查，更新 `STRETTO_RUN` 後：

```bash
ls -lh "$STRETTO_RUN/profiling"
head -n 8 "$STRETTO_RUN/profiling/batches.csv"
/opt/venv/bin/python -m json.tool "$STRETTO_RUN/profiling/summary.json"
```

輸出包含 prefill/decode 的 CPU/GPU 區間、forward/token 計數、CPU RSS/USS/PSS（權限允許時）、主機/cgroup RAM、GPU 記憶體與 utilization、模型 allocated/reserved/peak、cache 載入與實際程序 I/O、H2D/D2H bytes/時間、Linux minor/major faults。PyTorch trace 限正式 benchmark 每模型兩類路徑首個實際 batch，最多四份；沒走到的路徑不製造工作量補 trace。核對 trace_coverage／limitations，cpu_only 不能當有效 CUDA trace；區間 GPU 時間不全是純 kernel compute。[詳細定義](profiling.md)

首次 run 包含下載或 cache 建立，後續 profile 可能 warm，**不能直接用兩輪總時間差當 profiling overhead**。比較前另跑 warm run，再與相同資料/config/cache 狀態的 profile 對照並保留全部結果。

## 6. 可選：獨立執行 Nsight

前面 run/profile 通過後才做。這條 CUDA 容器路線直接裝固定版 CLI，無需 Docker 或 `STRETTO_NSYS_IMAGE`；該變數屬 VM 的 Docker launcher。下面使用與 `docker/Dockerfile.profiling` 相同的 NVIDIA 官方固定版套件，不重建推論依賴。是否允許 UM collector 仍受平台限制。

```bash
curl -fL --retry 3 -o /tmp/stretto-nsight-cli.deb https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/NsightSystems-linux-cli-public-2026.2.1.210-3763964.deb
sha256sum /tmp/stretto-nsight-cli.deb | tee /opt/stretto/nsight-package.sha256
apt-get update
apt-get install -y --no-install-recommends /tmp/stretto-nsight-cli.deb
nsys --version | tee /results/host-info/nsight-version.txt
cd /opt/stretto/source
/opt/venv/bin/python - <<'PY'
import json
from pathlib import Path
from reasondb.reproduction.nsight import check_nsys
report = check_nsys()
print(json.dumps(report, indent=2))
Path('/results/host-info/nsight-environment.json').write_text(json.dumps(report, indent=2))
PY
```

任一步失敗就先停。檢查版本應為 2026.2.1 系列，所需 capture／UM flags 齊全。`status --environment` 的限制要保留；不要自動加入 privileged，也不要把沒有資料當成 0 faults。

```bash
set -o pipefail
STRETTO_LOG="/results/terminal/profile-nsys-$(date -u +%Y%m%dT%H%M%SZ).log"
/opt/venv/bin/python /workspace/stretto-run/reproduce_in_container.py profile-nsys 2>&1 | tee "$STRETTO_LOG"
STRETTO_EXIT=${PIPESTATUS[0]}
printf 'profile-nsys exit code: %s\n' "$STRETTO_EXIT"
```

與 PyTorch profile 分開跑。等命令完全結束（含 report/SQLite 匯出），再做第 4 節檢查、查看 `profiling/nsight`、`profiling/cuda_page_faults.json` 與 services log。顯式 tensor 搬送不保證有 Unified Memory faults；沒有 UM table 不能證明零事件。此遠端平台支援狀態仍 pending。[NVIDIA 官方說明](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

## 7. 通過 100 篇後才擴大至 728 篇

先分析品質、峰值、cache 占用、trace，再決定是否跑；同機可跑，但完整資料會使用另一個 cache namespace，空間未必足夠。

```bash
df -h /data /results
du -sh /data/hf /data/cache /results
/opt/venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('/workspace/stretto-run/single_gpu_full.json')
if p.exists():
    raise SystemExit('Config exists; inspect it rather than overwrite')
c = json.loads(Path('/opt/stretto/configs/single_gpu.json').read_text())
c['name'] = 'llama8b_3b_rotowire_full_v1'
c['reports_limit'] = 728
p.write_text(json.dumps(c, indent=2) + '\n')
PY
/opt/venv/bin/python /workspace/stretto-run/reproduce_in_container.py inspect --config /workspace/stretto-run/single_gpu_full.json
```

inspect 確認 728 後：

```bash
set -o pipefail
STRETTO_LOG="/results/terminal/full-run-$(date -u +%Y%m%dT%H%M%SZ).log"
/opt/venv/bin/python /workspace/stretto-run/reproduce_in_container.py run --config /workspace/stretto-run/single_gpu_full.json 2>&1 | tee "$STRETTO_LOG"
STRETTO_EXIT=${PIPESTATUS[0]}
printf 'full run exit code: %s\n' "$STRETTO_EXIT"
```

重做第 4 節檢查。完整資料仍是同一固定 query，不等於論文完整 sweep。如要 full profile，保留相同 `--config`，將 action 換為 `profile`，另取 log 名稱。

## 8. 備份後再結束租借

遠端 Bash，等所有 run／collector 完成；壓縮檔也需要磁碟空間。

```bash
cp /workspace/stretto-run/reproduce_in_container.py /workspace/stretto-run/bootstrap_vast_cuda.sh /results/host-info/
if [ -f /workspace/stretto-run/single_gpu_full.json ]; then
  cp /workspace/stretto-run/single_gpu_full.json /results/host-info/
fi
STRETTO_ARCHIVE="/workspace/stretto-results-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
tar -czf "$STRETTO_ARCHIVE" -C / results
sha256sum "$STRETTO_ARCHIVE" > "$STRETTO_ARCHIVE.sha256"
printf '%s\n%s.sha256\n' "$STRETTO_ARCHIVE" "$STRETTO_ARCHIVE"
```

**本機 PowerShell**，替換 `REMOTE_ARCHIVE` 為上方完整遠端路徑，`ARCHIVE_NAME` 為實際檔名：

```powershell
scp -i "$env:USERPROFILE/.ssh/id_ed25519_stretto_vast" -o IdentitiesOnly=yes -P PORT USER@HOST:REMOTE_ARCHIVE .
scp -i "$env:USERPROFILE/.ssh/id_ed25519_stretto_vast" -o IdentitiesOnly=yes -P PORT USER@HOST:REMOTE_ARCHIVE.sha256 .
Get-FileHash -Algorithm SHA256 .\ARCHIVE_NAME.tar.gz
Get-Content .\ARCHIVE_NAME.tar.gz.sha256
tar -tzf .\ARCHIVE_NAME.tar.gz
```

核對 SHA256、可解壓、包含各輪 manifest/metrics/profiling/logs 後再結束租借。備份僅含 `/results`，不含 `.env`、私鑰、`/data` 權重或 KV cache；若要保留大型 cache 需另外備份。Destroy 會刪除 instance 的資料；Stop 仍有儲存費用。
