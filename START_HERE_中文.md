# 从这里开始

这是 Kaggle Playground Series S6E9“电动车购买预测”的完整项目。

## 先看结果

- 第一版原创模型：OOF ROC-AUC **0.946072**；Kaggle 公开榜 **0.94627**。
- 上一个已上传原创版本：OOF ROC-AUC **0.946298**；Kaggle 公开榜 **0.94640**。
- 最终原创模型：重复交叉验证 + 多视角秩融合，OOF ROC-AUC **0.946314**，Kaggle 公开榜 **0.94641**，5/5 个审计组均优于最强单模型。
- 同日 Top 10% 分界约为 **0.94650**；本次原创模型约处于前 16%–17% 分数区间，**尚未达到 Top 10%**。

## 推荐阅读顺序

1. `docs/中文解题思路.md`：从题目、验证、特征到集成的完整讲解。
2. `notebooks/01_ev_purchase_project.ipynb`：可运行、带输出的分析 Notebook。
3. `reports/final_report.md`：英文技术报告和实验表。
4. `src/evpurchase/`：模块化训练代码。
5. `submissions/submission_manifest.csv`：当前提交文件的成绩与哈希记录。

## 文件说明

- `submissions/submission_best.csv`：自己的模型提交文件。

## 申请材料中怎样写

可以写：

> 独立搭建了防泄漏的分层交叉验证与模块化树模型流水线，通过频率编码、数字位特征、嵌套交叉拟合目标编码、多尺度交叉特征、重复 10 折与浅层 20 折训练和多视角秩融合，将 OOF ROC-AUC 从 0.94607 提升到 0.94631；最终 Kaggle 公开榜为 0.94641，融合在 5/5 个审计组均优于最强单模型。
