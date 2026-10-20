# 租借前準備與 API 存取驗收

先完成這份清單，再租 GPU。這裡的 `access` 不需要 GPU；`doctor`、模型推論與效能驗收需要租借主機。

## 目前已核對的來源

- GitHub：<https://github.com/hazcashi/SemBench-Stretto_experiment>，分支 `main`。
- 程式版本：`436207bd211943709930577ec44895d8a1db0cd5`。
- 2026-10-20 從 GitHub 重新 clone，確認遠端／本機 commit 相同；在新 checkout 執行根目錄測試為 **32 passed、5 skipped**。這不是 CUDA 驗收。
- 在新 checkout 執行資料下載腳本，五份 Rotowire CSV 通過固定來源的 SHA256 驗證。
- `.env` 不受 Git 追蹤，也排除於 Docker build context；模型權重、cache、結果不會隨 GitHub clone 取得。
- `docker/image.txt` 仍指向舊 baseline 的依賴映像。詳細 profiling 必須依 [逐步指南](rented_gpu_quickstart.md) 建立新版程式映像，不能直接把舊 digest 當成新版。

後續若只有文件更新，文件 commit 會不同於上述已測試的程式版本。Docker 的來源標籤記錄的是建置時指定的版本。

## 1. 需要的帳號與憑證

| 項目 | 憑證／權限 | 程式實際用途 |
| --- | --- | --- |
| Hugging Face | 一個 `HF_TOKEN`；同帳號取得兩個 Llama repository 的讀取權限 | 下載固定 revision 的模型設定、tokenizer 與權重 |
| OpenAI API | 一個 `OPENAI_API_KEY`；有效額度與下列模型存取權 | 雲端 planner、metadata 與 embeddings |
| GitHub | 公開 repo clone 不需要 API key | 取得程式、設定與文件 |
| GHCR | 目前公開依賴映像可匿名拉取 | 取得已安裝推論依賴的 Docker 基底 |

OpenAI 的同一把 key 需要可以呼叫：

| Model ID | API | 用途 |
| --- | --- | --- |
| `gpt-4o-2024-11-20` | `POST /v1/chat/completions` | 自然語言 smoke 的查詢規劃／自我修正 |
| `gpt-4o-mini-2024-07-18` | `POST /v1/chat/completions` | metadata／資料準備相關推理 |
| `text-embedding-3-small` | `POST /v1/embeddings` | Stretto metadata 的文字向量 |

Llama 3B／8B 在租借 GPU 本地執行；此流程不呼叫 Hugging Face Inference Providers。文字 operator 使用的 `BAAI/bge-small-en-v1.5` 也是本地 embedding server，與上表的 OpenAI metadata embeddings 是兩個不同用途。

程式依據：`reasondb/reproduction/runner.py::check_access`、`reasondb/database/database.py`、`reasondb/reasoning/llm.py` 與 `reasondb/reasoning/embeddings.py`，均位於 `stretto-main/stretto-main/`。

## 2. Hugging Face：先取得兩個模型的授權

1. 登入自己的 Hugging Face 帳號。
2. 分別開啟 [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 與 [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)。依頁面要求接受授權／提交申請；兩個都要確認可以存取，不能只停在申請中。
3. 開啟 [Access Tokens](https://huggingface.co/settings/tokens)，建立本實驗專用 token，例如 `stretto-reproduction`。
4. 使用 `Read` token，或僅授權讀取上述模型的 fine-grained token。若使用 fine-grained token，必須包含自己已獲准存取的 gated repositories 讀取權限。這個專案不需要寫入模型 repository 的權限。
5. 將 token 存在本機 `.env` 的 `HF_TOKEN=` 後方。Token 所屬帳號必須是取得模型授權的帳號。

HF 的 gated access 是帳號層級；建立 token 本身不會自動取得模型授權。參見 [Gated models](https://huggingface.co/docs/hub/models-gated) 與 [User access tokens](https://huggingface.co/docs/hub/security-tokens)。

## 3. OpenAI：準備 API 平台額度與 key

1. 登入 [OpenAI API 平台](https://platform.openai.com/)，選擇自己可使用的 organization／project。可為實驗另建 project，方便查看用量。
2. 在 API 平台的 Billing 確認付款／可用額度。不要僅以已登入 ChatGPT 判斷 API 已開通。
3. 到 [API keys](https://platform.openai.com/api-keys) 建立這個 project 的 key，名稱可設 `stretto-reproduction`。
4. 若使用 Restricted key，須允許上表的 Chat Completions 與 Embeddings 請求；project 的模型／速率限制也須容許指定版本。實際可用性由下一步的小額請求驗證。
5. 將 key 存在 `.env` 的 `OPENAI_API_KEY=` 後方。之後在同一 project 的 Usage 查看實際費用；建立 key 不代表已有可用額度。

官方建立 key／環境變數說明見 [Developer quickstart](https://developers.openai.com/api/docs/quickstart)。實驗中的 API 呼叫會另外計費，Docker 建置與 CPU 單元測試不會呼叫這些 API。此專案沒有已驗證的全場 API 費用估計，也沒有程式層級的總金額上限。

**2026-10-20 查閱官方公告：上述兩個 GPT-4o snapshot 已 deprecated，預定於 2027-01-14 關閉。** Deprecated 不代表你的帳號現在一定無法使用；必須實際通過 `access`。若失敗，保留錯誤原因再決定如何處理，不應自行替換模型後稱為相同實驗。參見 [OpenAI deprecations](https://developers.openai.com/api/docs/deprecations)。

## 4. 填寫本機 `.env`

專案根目錄已有空白範本 `.env.example`。若 `.env` 尚未存在，複製一份；不要覆蓋已填妥的檔案。

```dotenv
HF_TOKEN=填入自己的HF_token
OPENAI_API_KEY=填入自己的OpenAI_API_key
STRETTO_IMAGE=stretto-profile:local
STRETTO_NSYS_IMAGE=stretto-nsys:local
STRETTO_DATA_DIR=./data
STRETTO_RESULTS_DIR=./results
```

以上中文是說明用占位文字，請換成真實值。不要把 key 貼到聊天、commit 或命令列參數。租借主機上也需要另行設定 `.env`；它不會從 GitHub 自動下載。借用電腦的實驗結束後，移除自己的憑證檔並撤銷不再使用的 token。

## 5. 在付 GPU 租金前做存取測試

使用 Docker Desktop 的 Linux engine；新版一般映像的建置方式見 [逐步指南第 3 步](rented_gpu_quickstart.md)。建置不需要 GPU，也不需要把 key 傳進 image build。

Windows PowerShell 在專案根目錄執行：

```powershell
# 確認已完成新版映像建置；檢查命令不會啟動 GPU。
docker image inspect stretto-profile:local --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
docker run --rm stretto-profile:local inspect

# .env 已填好後再做；會下載少量設定／tokenizer，並產生三次小額 API 請求。
New-Item -ItemType Directory -Force -Path data, results | Out-Null
docker run --rm --init --env-file .env `
  --mount "type=bind,source=$($PWD.Path)\data,target=/data" `
  --mount "type=bind,source=$($PWD.Path)\results,target=/results" `
  stretto-profile:local access
```

Linux／WSL 可用 `bash scripts/reproduce.sh inspect` 及 `bash scripts/reproduce.sh access`，但 `.env` 必須設定新版 `STRETTO_IMAGE`。

`access` 成功必須回報 `status: passed`。它實際讀取兩個指定 revision 的 `config.json`／`tokenizer.json`，呼叫兩個指定 GPT snapshot 及一個 embedding model；不下載完整權重、不做模型推論，也不能證明雙模型能同時放入 VRAM。

錯誤需分別處理：HF 401／403 檢查 token、模型授權與 scope；OpenAI 401 檢查 key；403／model-not-found 檢查 project 權限與模型可用性；429 需查看錯誤內容，區分額度不足與速率限制。不要將全部錯誤視為沒有錢。

## 6. 租借條件與仍需現場驗收的項目

租借方案需能提供 Linux x86_64、單張約 48GB NVIDIA GPU、可用 Docker daemon、host NVIDIA driver 與 NVIDIA Container Toolkit。目標為 Ampere／Ada；建議 RAM 64GB。映像準備完後，資料掛載位置仍須至少 100GiB 可用，Docker layers、權重、KV caches 與 trace 另計空間。僅提供 SSH 或 GPU notebook 不能保證允許這個 Docker 工作流程。

租借前先確認供應商是否允許 Docker 與 Nsight CUDA tracing；不預設能使用 privileged container。平台是否支援 Unified Memory fault collector，要到現場檢查。

只有以下項目都完成，才代表租借前準備通過：

- 新版一般映像建置、無 GPU `inspect` 通過。
- 兩個 Llama 模型與三個 OpenAI model 的 `access` 全部通過。
- 若要跑 `profile-nsys`，獨立 Nsight 映像也已建好，CLI 能啟動；GPU collector 仍待現場驗收。
- 租借方案符合 GPU、RAM、磁碟、Docker 條件；帳號與 SSH 連線方式已準備。

GPU 的 CUDA／FlashAttention kernel、微型 Llama 測試、雙模型共存、100 篇完整執行、代表 batch trace、Unified Memory collector 與 profiling overhead，在真正的租借 GPU 上另行驗收。目前 CPU 測試通過不能代替這些項目。
