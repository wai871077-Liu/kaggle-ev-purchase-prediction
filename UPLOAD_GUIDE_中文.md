# Kaggle 与 GitHub：分别提交什么

## Kaggle 比赛提交

只上传 `submissions/submission_best.csv`，在比赛页面点击 **Submit Prediction**。

当前最终文件的 SHA-256 为
`616a1d0dedfb6c72b5dbd8cdda4926960cb9cbce0f6f3f4b1cd91d190545f82b`；
Kaggle 上传名是 `submission_original_v4.csv`，已于 2026-09-27 完成并核实公开榜 **0.94641**。

它应当有两列 `id,Will_Buy_EV`，包含 286,571 行测试集预测。不要上传整个文件夹、Notebook、训练集或标签文件。上传前运行 `scripts/validate_submission.py`，核对文件结构与 `submissions/submission_manifest.csv` 中的 SHA-256。

比赛页面：<https://www.kaggle.com/competitions/playground-series-s6e9>

若当前版本已经提交且哈希相同，无须重复上传。比赛截止后的最终私榜成绩与当前公开榜成绩可能不同。

## GitHub 项目展示

仓库：<https://github.com/wai871077-Liu/kaggle-ev-purchase-prediction>

公开内容包括：

- `src/`、`scripts/`、`tests/`：模块化代码、复现和校验工具。
- `notebooks/`：执行过的分析 Notebook。
- `reports/`、`docs/`：实验指标、图表、技术报告和中文学习说明。
- `README.md`、`START_HERE_中文.md`、本文件、`CITATIONS.md`、`LICENSE`。
- `requirements.txt`、`pyproject.toml`、`.gitignore`、`data/README.md`。
- `submissions/submission_manifest.csv`：当前预测文件的版本、成绩与校验信息。

公开仓库已配置 `.gitignore`，不会上传比赛原始数据、中间预测数组、最终提交 CSV、虚拟环境和缓存。这些内容保留在桌面完整学习包中，而 GitHub 只展示可复现代码、评估结果与文档。

本地完整学习包还保留原始 CSV、预测数组和最终提交 CSV。这些文件不随公开仓库上传；`.gitignore` 已排除它们。也不要上传虚拟环境、凭据、缓存或系统文件。

桌面的 `Electric vehicle` 是个人完整学习包，不能不加筛选地整体拖进 GitHub。
