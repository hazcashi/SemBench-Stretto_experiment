# 從租借主機到最小實驗

## 租借前

1. Hugging Face 分別申請 `meta-llama/Llama-3.1-8B-Instruct` 與 `meta-llama/Llama-3.2-3B-Instruct` 的 gated access，接受授權，建立 read token。
2. 準備 OpenAI API key 與可用額度。原始程式使用 `gpt-4o-2024-11-20`、`gpt-4o-mini-2024-07-18`、`text-embedding-3-small`，不只在 query planning 使用 API。
3. 確認供應商給的是可執行 Docker 的 Linux 主機，NVIDIA driver、NVIDIA Container Toolkit 與 Docker daemon 可用；SSH 本身不能證明具備這些條件。不在 GPU container 內假設 Docker-in-Docker。
4. 選約 48GB GPU、建議 RAM 64GB，最小部署預留至少 100GiB 可用磁碟。完整 sweep 另估。首次模型下載與 cache 建立仍需時間；預建 image 只省去環境安裝。

## 使用已發佈映像

```bash
git clone --recurse-submodules https://github.com/hazcashi/SemBench-Stretto_experiment.git
cd SemBench-Stretto_experiment
cp .env.example .env
# 編輯 .env：HF_TOKEN、OPENAI_API_KEY、STRETTO_IMAGE=ghcr.io/...@sha256:...
bash scripts/reproduce.sh doctor
bash scripts/reproduce.sh run
```

`STRETTO_IMAGE` 必須是實際建成的 tag/digest。目前是否已發佈見 experiment_log；不要把示意 placeholder 當可用映像。發佈後可把 digest 寫入 `docker/image.txt`，供所有 checkout 預設使用。GHCR 新套件預設 private；可在 package settings 調整為 public，或在租借主機用具 `read:packages` 權限的 token 先執行 `docker login ghcr.io`，不要將密碼放進腳本或 Git。GitHub 網站／Git 的登入不等同有 GHCR 權限，參見 [GitHub 官方說明](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)。

doctor：確認單張可見 GPU、CUDA、約 48GB VRAM、100GiB free、兩模型 config/tokenizer 權限，以及上述三種 OpenAI model 的實際小額存取；會產生少量 API 費用。沒有 token 時會立即指出缺少的變數，不下載權重。

run：重新做 preflight，下載固定 revision 的 safetensors，生成固定資料子集，先各別測試 3B 和 8B，再啟動雙模型及文字 embedding，透過 upstream prepare 生成兩種 cr0.8 caches。之後跑自然語言 smoke query、Rotowire 原固定 query，以及 vanilla 8B silver reference。benchmark 保留原固定 logical plan；smoke 才測真實 NL planner。原固定 query 集只有一個，不是論文 sweep。

租借前可在本機執行 `bash scripts/reproduce.sh access`，不需要 GPU，僅測試 HF 與 OpenAI 存取，不下載模型權重；會有小額 API 呼叫。

Windows PowerShell 不使用 WSL 時，也可直接用 Docker Desktop 的 Linux engine 做同樣存取檢查；先填妥 `.env`，並使用發佈後的 digest：

```powershell
$image = (Get-Content docker/image.txt -Raw).Trim()
New-Item -ItemType Directory -Force -Path data, results | Out-Null
docker run --rm --init --env-file .env --mount "type=bind,source=$($PWD.Path)\data,target=/data" --mount "type=bind,source=$($PWD.Path)\results,target=/results" $image access
```

第一次可先執行 `bash scripts/reproduce.sh inspect`，不用 GPU 或 token 即可核對 operator 路由。run 的 smoke 是已知答案測試；model 回傳不確定或錯誤就失敗，不宣稱後端正常。

## 自行建置與發佈

```bash
git submodule update --init --recursive
docker build --platform linux/amd64 --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" -t stretto-baseline:local .
STRETTO_IMAGE=stretto-baseline:local bash scripts/reproduce.sh inspect
# 通過後發佈；先登入 GHCR
docker tag stretto-baseline:local ghcr.io/hazcashi/sembench-stretto_experiment:YOUR_COMMIT
docker push ghcr.io/hazcashi/sembench-stretto_experiment:YOUR_COMMIT
```

保留 push 回傳的 digest，使用 `ghcr.io/hazcashi/sembench-stretto_experiment@sha256:...`。也可手動觸發 GitHub Actions 的 Build and publish baseline image；輸出 artifact `immutable-image-reference/image.txt`。沒有自動 GPU CI。

Dockerfile 以 Python 3.12、CUDA 12.8.1、torch 2.9.1+cu128、torchvision 0.24.1+cu128、FA2 2.8.3.post2 建置。依賴 lock 固定 resolver 結果及 hashes。此 FA2 release 無 torch2.9 的對應 wheel，會以單一編譯工作編譯，以免用 ABI 不相容的 torch2.8 wheel。使用 upstream 的 `FLASHATTENTION_DISABLE_BACKWARD` 編譯選項，保留全部 forward head dimensions，只供推論；不適用需要 attention autograd 的模型訓練。此選項記錄於 run manifest 的 flash_attention_build。支援 Ampere/Ada 的編譯架構；其他 GPU 世代要另外驗證。

本地 Linux/Windows Docker Desktop 皆可 build，不需要 48GB GPU；需要足夠 RAM/磁碟及下載時間。建置 log 與成功狀態見 experiment_log。

## 持久資料與結果

- `data/hf/`：固定 model revision 的 Hugging Face cache。
- `data/cache/`：Stretto metadata、KV cache、prompt cache，跨 run 保留。
- `data/<profile-name_config-hash>/subset/`：固定資料子集及來源 hash。
- `results/<UTC timestamp>-<random id>/`：manifest、doctor、queries、metrics CSV、events JSONL、resource samples、backend logs／trace 或 failure.txt。
- `results/logging/<run-id>/`：upstream FileLogger 的完整 operator／planner log。

映像內程式與 config 固定；修改 Git checkout 的程式或 configs 後，需重建映像才會生效。profile 內容 hash 納入 Stretto cache namespace，避免更換模型 revision 或子集設定後讀到舊 metadata 或 KV。

結果 cache 停用，每次 run 用新 ID；KV 與雲端 prompt cache 可能為 warm，manifest 記錄事前 cache 狀態。不要把重跑當成 cold run。要測 cold/warm，使用不同 `STRETTO_DATA_DIR` 指向新的空目錄，保留原資料即可。

結果 precision/recall 預設對 vanilla 8B silver 計算，並非人類標註或論文 70B reference。模型改變後不保證 0.9 目標一定可達；輸出保留 optimizer guarantee 判定。子集的 rowid 改變，不套用原始人類 labels。

即使原引擎因某個 operator 失敗而使用 fallback 完成查詢，只要 backend request log 存在 HTTP error 或 cache generation error，此 runner 仍將 run 標成 failed，保留輸出供診斷，不把這次執行當成有效 baseline。

啟動器只終止自己建立的 process groups，不使用 stop_servers 掃描整台主機；port 被占用會拒絕啟動。缺少 key、OOM、readiness timeout 均保留 failure 與非零退出碼。預設不使用 privileged 容器。
