# Upstream 與來源版本

- 專案 remote：`https://github.com/hazcashi/SemBench-Stretto_experiment.git`，審查開始時無任何 commit。
- 比對 upstream：`https://github.com/DataManagementLab/stretto.git`，`6a540eab0581fa7e4b2605421839bfc3dde77eae`。
- 下載 archive 真正起源 SHA 未知；比對 commit 不能當成 archive provenance 的證明。修改前逐檔結果保存在 `source_provenance.json`，683 個檔案一致、7 個 CSV 檔案不同、無遺失；文字以 LF 正規化、二進位圖片以原始 bytes 比較。相關 Python 檔案全數一致。
- kvpress：`https://github.com/GabrieleSanmartino/kvpress.git`，`81c4c78bdcd7d4c2f18e6878e112a0599a953b43`。
- palimpzest：`https://github.com/mitdbg/palimpzest.git`，`af44b09b7a6df302863e8fa28eb5fe9709e40d09`。保存 source，不安裝其整套環境；其 pandas~=2.1.1 與此 baseline 的 pandas>=2.3.3 不一致，也非 Rotowire 路徑需求。

外層 `.gitmodules` 才是 fresh clone 的有效 submodule 定義；原 archive 內的 `.gitmodules` 保留供比對。Windows 上 palimpzest 含尾隨句點的檔名，局部 sparse checkout 省略 testdata；Linux 上可正常 recursive clone。

Upstream：建議 Python 3.12；requirements 指定 transformers==5.2.0、tokenizers==0.22.2、duckdb==1.3.0；pyproject 要求 torch>=2.9.1。文字預設 Llama 8B small :5010、70B large :5012；scripts 預設 8B GPU 3、70B GPU 0/1/2、embedding GPU 3。文字 launch script 亦啟動 image similarity :5005 與 text embed :5324。

新增 profile 保留上述 registry 角色與預設，另註冊 3B :5014／kv3B；profile 內才指定 8B large、3B small，不把 kv70B 誤當 8B cache 名稱。

本機審查環境：Windows、Python 3.13.2（非正式執行環境）、RTX 4060 Laptop 8188MiB、driver 592.82、Docker Desktop Linux engine 29.5.2。正式映像 Python 3.12；精確執行環境保存於每個 run 的 pip-freeze 與 manifest。
