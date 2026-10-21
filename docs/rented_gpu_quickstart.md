# 從 GitHub 到租借 GPU：100 篇與 profiling

**2026-10-21 已發佈新版一般映像。租到 Vast VM 後，優先使用 [主機操作指令手冊](vast_vm_runbook.md)**，直接 pull 固定 digest；不需要在付 GPU 租金時重新建置。下方保留從原始碼建置的流程，供需要更新程式時使用。

本指南適用這次新增的詳細 profiling 原始碼。CPU 測試已完成；48GB GPU、雙模型、真實查詢、CUDA trace、Nsight collector 與 overhead 仍待現場驗收，不保證第一次執行即成功。

## 1. 租借前準備

- 能執行 Docker 的 Linux x86_64 主機；先以專案既定的 Ampere/Ada 約48GB NVIDIA GPU 為目標。
- 建議主記憶體64GB。資料掛載位置在每次 doctor/run/profile 前都須至少100GiB可用，不能只租總容量100GB；Docker、模型權重、KV caches 與 traces 另占空間。
- 主機 NVIDIA driver、Docker daemon、NVIDIA Container Toolkit 已備妥。只有可登入的 GPU 容器不等於能在其中執行 Docker。
- HF 帳號已取得 meta-llama/Llama-3.1-8B-Instruct、meta-llama/Llama-3.2-3B-Instruct 兩者權限，準備 read token。
- 準備有額度的 OpenAI API key；planner、metadata 與 embeddings 會呼叫 API，access/doctor 也會產生少量費用。

本機只需 SSH 操作，不需安裝專案 Python/CUDA 套件或下載權重。

## 2. 在租借主機 clone

```bash
nvidia-smi
docker info

git clone --recurse-submodules https://github.com/hazcashi/SemBench-Stretto_experiment.git
cd SemBench-Stretto_experiment
git rev-parse HEAD
```

若先前已 clone，先確認沒有自己的未提交修改，再以 `git pull --ff-only` 和 `git submodule update --init --recursive` 更新。不使用 reset/clean 覆蓋個人資料。

長時間工作可在主機使用 `tmux new -s stretto`；中途按 Ctrl+B、再按 D 可離開 session，之後 `tmux attach -t stretto` 回到原工作。

## 3. 沿用依賴，建置新版程式映像

`docker/image.txt` 的舊公開映像包含已建好的 torch/FlashAttention/kvpress 環境，**不包含這次新的詳細 profiling**。使用 source-update Dockerfile 複製新版程式並重跑檢查，不必再次編譯 FlashAttention；本次沒有變更依賴輸入。

```bash
BASE_IMAGE="$(tr -d '\r\n' < docker/image.txt)"
docker build --platform linux/amd64 \
  -f docker/Dockerfile.source-update \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t stretto-profile:local .
```

建置會抓取/校驗五份公開 Rotowire CSV、執行依賴與匯入檢查、inspect、CPU tests；不需要 HF/OpenAI secrets。建置失敗先停止並保存輸出，不把不完整映像拿去跑。未來若更新 requirements、kvpress 或主 Dockerfile，應改用完整建置，不沿用這個依賴捷徑。

## 4. 填寫 .env

```bash
cp .env.example .env
nano .env
```

```dotenv
HF_TOKEN=填入你的HF_token
OPENAI_API_KEY=填入你的OpenAI_API_key
STRETTO_IMAGE=stretto-profile:local
STRETTO_NSYS_IMAGE=stretto-nsys:local
STRETTO_DATA_DIR=./data
STRETTO_RESULTS_DIR=./results
```

以 persistent disk 的絕對路徑覆寫 DATA/RESULTS 也可以。不要提交填好的 .env。使用新版 profiling 時不要把 STRETTO_IMAGE 留白，否則啟動器會選舊 release digest。

## 5. 前置驗收與微型 CUDA 測試

```bash
bash scripts/reproduce.sh inspect
bash scripts/reproduce.sh access
bash scripts/reproduce.sh doctor

docker run --rm --gpus device=0 \
  --entrypoint python stretto-profile:local \
  -m pytest /opt/stretto/tests/test_profiling.py \
  -k cuda_real_tiny_llama -q
```

每一步成功後才進下一步。doctor 會實際檢查一張可見GPU、顯存、磁碟、CUDA tensor、BF16 FlashAttention kernel、HF config/tokenizer 與 API 權限。微型測試不下載權重、不呼叫 OpenAI，驗證 vanilla/cached generation 開關 instrumentation 後 token 一致；這項在 GPU 主機應 passed，而非因沒有CUDA而 skipped。

## 6. 跑100篇 baseline，再跑詳細 profiling

```bash
bash scripts/reproduce.sh run
```

腳本會下載固定模型 revision、選前100篇報導及關聯資料、依序驗證3B/8B單獨與共同駐留、準備KV caches、跑自然語言smoke、固定Rotowire查詢與vanilla 8B reference。無須手動啟動 upstream servers。

檢查該次 results/<run-id>/manifest.json 為 completed、backend_errors 為空；有 failure.txt 時先診斷。completed 只表示流程完成，不表示 precision/recall 已達0.9，品質另看 metrics.csv。

成功後執行：

```bash
bash scripts/reproduce.sh profile
```

這會重新跑同一套實驗，新增 prefill/decode、記憶體、搬送、缺頁數值與限量 PyTorch GPU trace。優先讀 profiling/batches.csv、summary.json、summary.csv、resources.jsonl 和 traces/。模型/查詢路徑未被選到時，不會補造 trace。

## 7. 額外量測 CUDA Unified Memory faults

```bash
docker build --platform linux/amd64 \
  -f docker/Dockerfile.profiling \
  --build-arg BASE_IMAGE=stretto-profile:local \
  -t stretto-nsys:local .

bash scripts/reproduce.sh profile-nsys
```

Nsight 選配映像也尚待實際建置驗收。此模式再次跑實驗，與 PyTorch Profiler 分開使用；保存 nsight_environment.json、nsight/*.nsys-rep、SQLite、cuda_page_faults.json。平台不支援或缺少權限時保存限制/錯誤，不自動使用 privileged。

程式沒有改成 Unified Memory，因此沒有UM事件是可能的；缺少collector資料也不能當作0次faults。CUDA/UM trace 是代表batch的取樣，不是整場benchmark。

## 8. 備份結果

在租借主機、專案根目錄：

```bash
mkdir -p results
git rev-parse HEAD > results/source_commit.txt
tar -czf ../stretto-results.tar.gz results
```

在本機 PowerShell，將 PORT/USER/HOST/REMOTE_PROJECT 換成實際SSH資訊和主機上的專案絕對路徑：

```powershell
scp -P PORT USER@HOST:REMOTE_PROJECT/../stretto-results.tar.gz .
```

備份所有 run 目錄（包含失敗紀錄）及 results/logging。若資料掛載在專案外的絕對路徑，請對實際 STRETTO_RESULTS_DIR 打包；上述預設指令只包含 ./results。大型 data/ 可保留在主機 persistent disk，避免重下權重/重建 cache。新 run 會重用應用與OS cache，不能直接視為 cold/warm 公平對照。

量測定義與限制見 [profiling.md](profiling.md)。目前工作量仍是縮小模型/資料的一題固定 benchmark，並非論文完整 sweep。
