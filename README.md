# Stretto 單卡文字 baseline

目標：單張約 48GB NVIDIA GPU，Llama-3.1-8B（large）＋Llama-3.2-3B（small），原始 Stretto / kvpress 後端。

這是重設模型與環境後的最小復現，尚未代表論文完整結果。查詢規劃、metadata 與部分程式生成仍會使用 OpenAI API。

**2026-10-27 換機備份：100／728 篇 profile 已完成，結果與 10 張圖表隨 Git 保存。請先看 [結果與還原說明](stretto-results/README.md)、[中文分析報告](stretto-results/analysis/report.md) 或 [圖表 PDF](stretto-results/analysis/charts.pdf)。** 原始 249 個檔案完整放在 ZIP，clone/pull 後以 `python scripts/restore_results.py` 解壓並驗證；閱讀 HTML／PDF 不需要 GPU、Docker 或憑證。

**重跑實驗：請從 [Vast 預建 image 與一鍵 100＋728 篇流程](docs/vast_container_runbook.md) 開始。** 本次在預建 image 上部署了程式修正並完成 GPU benchmark；這些修正與啟動腳本現已納入原始碼，舊 GHCR image 尚未重新發佈，不可假設只拉舊 image 就已有修正。

**Vast 自訂 `Stretto SSH` 模板請依照 [一般容器操作手冊](docs/vast_container_runbook.md)**；Ubuntu VM 才使用 [VM 操作手冊](docs/vast_vm_runbook.md)。兩條路線的 SSH、啟動與資料路徑不同，請勿混用 Docker 命令。手冊保留初始交接與修正歷程，最新成功結果及換機步驟以上方結果說明為準。

本次 GPU 實驗已驗證兩模型、微型 CUDA、固定 benchmark 與每輪四份含 CUDA kernels 的 PyTorch trace；本機 CPU 回歸為 58 passed、1 skipped。額外自然語言 smoke 改為選跑，Nsight／Unified Memory faults 與 profiling overhead 尚未驗收。舊 baseline 保留於 [依賴基底 digest](docker/image.txt)，該舊映像不含新版詳細 profiling。

先完成 [租借前準備與 API 存取驗收](docs/pre_rental_checklist.md)，確認模型授權、API 額度與新版映像。從GitHub到租借主機的完整操作見 [租借GPU逐步指南](docs/rented_gpu_quickstart.md)；量測定義見 [profiling](docs/profiling.md)。另有 [部署指南](docs/deployment.md)、[驗證紀錄](docs/experiment_log.md)、[架構審查](docs/reproduction_plan.md)。舊版映像與CI證據見 [release.json](docker/release.json)。

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
