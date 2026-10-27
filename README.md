# Stretto：Rotowire 728 篇／60 題

單張約 48GB NVIDIA GPU，Llama-3.2-3B（small）＋Llama-3.1-8B（large），沿用 Stretto optimizer 與 kvpress backend。

**本次操作只看 [Vast 操作手冊：728 篇／60 題](docs/full_rotowire_vast.md)。** 包含 Vast image、GitHub clone／pull、`.env`、資料下載、正式啟動、接續及備份指令。

本次是 Rotowire 全資料與上游產題器的 60 題，使用 batch=1、precision／recall=0.9、cr=0.8＋vanilla 候選；不包含 Movies／Email 或整篇論文的全部實驗。Operator configuration／metadata 等仍使用 OpenAI API。

原始碼位於 `stretto-main/stretto-main/`，資料來源校驗檔位於 `configs/source_provenance.json`。模型權重、下載資料、KV cache、`.env` 不進 Git。GitHub 存放程式；GHCR 的預建 image 提供執行環境。

先前單題實驗的結果備份仍保存在 [stretto-results](stretto-results/README.md)，與本次 60 題的新結果分開。本次完整 GPU 執行仍待租借主機驗收。
