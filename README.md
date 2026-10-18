# Stretto 單卡文字 baseline

目標：單張約 48GB NVIDIA GPU，Llama-3.1-8B（large）＋Llama-3.2-3B（small），原始 Stretto / kvpress 後端。

這是重設模型與環境後的最小復現，尚未代表論文完整結果。查詢規劃、metadata 與部分程式生成仍會使用 OpenAI API。

先閱讀 [部署指南](docs/deployment.md)、[驗證紀錄](docs/experiment_log.md)、[架構審查](docs/reproduction_plan.md)。預建映像的可用狀態以驗證紀錄為準；只有建置及發佈成功後才有可下載的 GHCR digest。

```bash
git clone --recurse-submodules https://github.com/hazcashi/SemBench-Stretto_experiment.git
cd SemBench-Stretto_experiment
cp .env.example .env
# 填入兩個 token 與已發佈的 STRETTO_IMAGE=digest
bash scripts/reproduce.sh doctor
bash scripts/reproduce.sh run
```

請先取得兩個 Llama checkpoint 的 Hugging Face 權限與 OpenAI API 額度。映像不含權重、token 或個人資料；公開 Rotowire 資料於 image build 時從固定 upstream commit 下載並驗證 SHA256。執行資料與輸出掛載在 `data/`、`results/`，不進 Git。

原始碼保留在 `stretto-main/stretto-main/`，upstream 入口預設仍為原論文配置。單卡實驗只用 `scripts/reproduce.sh` 與新增的 `reasondb.reproduction`。
