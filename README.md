# Stretto 單卡文字 baseline

目標：單張約 48GB NVIDIA GPU，Llama-3.1-8B（large）＋Llama-3.2-3B（small），原始 Stretto / kvpress 後端。

這是重設模型與環境後的最小復現，尚未代表論文完整結果。查詢規劃、metadata 與部分程式生成仍會使用 OpenAI API。

舊 baseline 的公開 GHCR 映像保留於 [固定 digest](docker/image.txt)，可作為已建好 torch/FlashAttention/kvpress 的依賴基底。**最新的詳細 profiling 在原始碼中，須先建置新版程式映像；舊 digest 不含新功能。** 新版根目錄 CPU 測試於本機 Python3.12 為32 passed、5 skipped；Docker建置、HF/API、48GB GPU雙模型、query/benchmark與Nsight仍需驗收。

從GitHub到租借主機的完整操作見 [租借GPU逐步指南](docs/rented_gpu_quickstart.md)；量測定義見 [profiling](docs/profiling.md)。另有 [部署指南](docs/deployment.md)、[驗證紀錄](docs/experiment_log.md)、[架構審查](docs/reproduction_plan.md)。舊版映像與CI證據見 [release.json](docker/release.json)。

```bash
git clone --recurse-submodules https://github.com/hazcashi/SemBench-Stretto_experiment.git
cd SemBench-Stretto_experiment
docker build --platform linux/amd64 -f docker/Dockerfile.source-update \
  --build-arg BASE_IMAGE="$(tr -d '\r\n' < docker/image.txt)" \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" -t stretto-profile:local .
cp .env.example .env
# 填入 HF_TOKEN、OPENAI_API_KEY，並設定 STRETTO_IMAGE=stretto-profile:local
bash scripts/reproduce.sh doctor
bash scripts/reproduce.sh run
bash scripts/reproduce.sh profile
```

請先取得兩個 Llama checkpoint 的 Hugging Face 權限與 OpenAI API 額度。映像不含權重、token 或個人資料；公開 Rotowire 資料於 image build 時從固定 upstream commit 下載並驗證 SHA256。執行資料與輸出掛載在 `data/`、`results/`，不進 Git。

原始碼保留在 `stretto-main/stretto-main/`，upstream 入口預設仍為原論文配置。單卡實驗只用 `scripts/reproduce.sh` 與新增的 `reasondb.reproduction`。
