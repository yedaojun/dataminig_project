# Group 19 项目交接：A、B、C 已完成内容与 D 工作流

**交接对象：D 工作流负责人。** 本文件以仓库当前的 A/B/C 代码和实际实验结果为准。以下路径均相对仓库根目录。C 的实现目前在本地分支 `group19-c-supervised`，提交 `2adbf27`；**截至本交接文件写作时尚未推送 GitHub**，因此 D 同学需要收到该分支或包含此文件的仓库副本，才能使用 C 的产物。B 的旧文档中“等待 C 交付”的描述属于撰写时状态，本文件更新了这部分状态。

## 一、项目目标与统一口径

数据集是 UCI AI4I 2020 Predictive Maintenance，包含 10,000 条合成记录。主任务是预测 `Machine failure`（二分类）；D 的主要任务是诊断五个可重叠的故障模式 `TWF, HDF, PWF, OSF, RNF`，并研究传感器噪声下的鲁棒性。它不是设备剩余寿命预测，也不能把基准结果当作真实工厂部署性能。

所有工作流共用 A 冻结的训练 / 验证 / 测试划分：**7,000 / 1,500 / 1,500**，按 `Machine failure` 分层，随机种子 42。正类为原始 `Machine failure == 1`；五个模式标签保持原值，不把标签冲突记录悄悄删掉或改写。模型输入只允许气温、工艺温度、转速、扭矩、刀具磨损、`Type`，以及 A 定义的可选三个工程特征。`UDI`、`Product ID`、总故障标签和五个模式标签都不得作为预测输入。

| 划分 | 行数 | 总故障 | TWF | HDF | PWF | OSF | RNF |
|---|---:|---:|---:|---:|---:|---:|---:|
| 训练 | 7,000 | 237 | 33 | 73 | 72 | 71 | 13 |
| 验证 | 1,500 | 51 | 5 | 24 | 10 | 12 | 4 |
| 测试 | 1,500 | 51 | 8 | 18 | 13 | 15 | 2 |

模式可重叠，不能把各列相加当作故障总数。A 找到 27 条总故障标签与模式标签不一致的记录：9 条总故障为 1 而五个模式全为 0；18 条总故障为 0 却至少有一个模式为 1。D 必须明确说明诊断指标的分母和“正常”的定义。特别是 RNF 测试仅有 2 个模式正例，结论很不稳定。

## 二、A：数据治理与公共接口，已经完成

A 已完成原始数据审计、标签冲突记录、固定索引、公共预处理与特征工程、配置和复现检查。关键文件：

- `Group19_A/data/raw/ai4i2020.csv`：原始数据，禁止修改字节；`Group19_A/data/splits/indices.npz` 和 `manifest.json`：冻结索引与数据指纹。
- `Group19_A/pipeline.py`：唯一的公共数据入口，提供 `load_split`、`make_preprocessor` 和 `features`。
- `Group19_A/config.json`：共同随机种子、5% 正常样本目标假阳性率、噪声级别及重复种子。
- `Group19_A/results/split_counts.csv`、`label_inconsistencies.csv`：上表与标签冲突的原始依据；`Group19_A/HANDOFF.md`：完整 A 接口说明。

从**仓库根目录**调用：

```python
from Group19_A.pipeline import load_split, make_preprocessor

X_train, y_train, Y_train = load_split("train", engineered=True)
X_val, y_val, Y_val = load_split("validation", engineered=True)
pre = make_preprocessor(engineered=True, scale=False)
Z_train = pre.fit_transform(X_train)
Z_val = pre.transform(X_val)
# Y_train / Y_val 为五列模式标签，行索引和 X、y 对齐。
# 冻结全部模型和阈值以后，才使用 allow_test=True 读取 test。
```

`engineered=False` 只有六项原始输入；`True` 加入 `temperature_difference`（工艺温度减气温）、`mechanical_power`（扭矩 × 转速 × 2π/60）、`wear_load`（磨损 × 扭矩）。预处理器**只在训练集 fit**。`load_split` 保留原始数据行索引，按位置索引时要使用 `.iloc`。A 的 `config.json` 中 `test_model_evaluation_authorized` 只是旧状态记录，不能替代团队的“先冻结、后测试”实验程序。

## 三、B：无标签异常检测，已经完成

B 已实现 Isolation Forest 与 novelty 模式 LOF。模型拟合只使用训练特征；有标签验证集用于候选选择和阈值校准。连续分数越高越异常。B 还做了散热分数与工程特征 LOF 的融合扩展、最终测试、PR/ROC 图和错误案例分析。主要入口与说明在 `Group19_B/README.md`、`Group19_B/BC_HANDOFF.md`、`Group19_B/误差分析报告.md`。

| B 模型 | 特征组 | 测试 AP | 测试 F1 | 检出 TP / 51 | 误报 FP / 1,449 |
|---|---|---:|---:|---:|---:|
| Isolation Forest | raw | 0.164 | 0.143 | 5 | 14 |
| LOF | raw | 0.223 | 0.276 | 16 | 49 |
| Isolation Forest | engineered | 0.335 | 0.350 | 14 | 15 |
| LOF | engineered | 0.325 | 0.279 | 12 | 23 |
| 散热 + LOF 融合扩展 | heat_fusion | 0.581 | 0.538 | 21 | 6 |

主方法对照保留两个 `raw` B 基线；工程组与融合分别标识，不混成同一输入条件。B 的测试逐行预测与指标位于 `Group19_B/results/final_test/<目录哈希>/predictions.csv` 和 `metrics.csv`；`Group19_B/results/frozen/CURRENT` 指向冻结模型。B 的误差分析发现两种原始基线共同漏掉 35/51 个故障；融合主要改善散热故障，对其他模式的覆盖未必改善。D 做模式解释时可以复用其逐行结果，但诊断模型仍须独立训练和评估。

## 四、C：监督分类与特征分析，已经完成

C 已实现 Logistic Regression、Random Forest、XGBoost，分别比较普通与类别加权训练、原始与工程特征。每个模型 × 权重 × 特征组给 3 个预设候选，共 36 次拟合；在验证集按 AP 选参、按正类 F1 选报警阈值，平局取更高阈值，然后冻结模型再评价测试。C 的 `score` 是正类概率，报警规则 `score >= threshold`。路径：`Group19_C/run.py`、`config.json`、`README.md`、`C_REPORT.md`。

| C 模型（普通训练） | 原始输入测试 AP / F1 | 工程输入测试 AP / F1 |
|---|---:|---:|
| Logistic Regression | 0.425 / 0.429 | 0.475 / 0.449 |
| Random Forest | 0.756 / 0.690 | **0.917 / 0.882** |
| XGBoost | 0.802 / 0.759 | 0.849 / 0.851 |

类别加权在本实验中没有稳定改善验证 AP 或测试 F1，不能假定“加权一定更好”。工程特征对树模型的提升很大，但这些特征接近合成数据的生成规律，不能推广为真实设备的普遍效果。C 已输出 12 组验证/测试逐行预测、B/C 并列表、同参数配对的特征/权重消融、验证集置换重要性和三张图。重要路径：

- `Group19_C/results/freeze_manifest.json`、`frozen_models.joblib`：冻结参数、阈值、数据和代码指纹、已拟合模型。跨 Python / scikit-learn / XGBoost 版本加载 joblib 可能不兼容；必要时按 `Group19_C/README.md` 重训。
- `Group19_C/results/test_predictions.csv`、`test_metrics.csv`：12 组测试预测与指标；`bc_comparison.csv`：B/C 统一对照。
- `Group19_C/results/permutation_importance.csv`、`matched_feature_ablation.csv`：特征分析与同参数消融。
- `python -m Group19_C.check_results`：核对 C 的分数、阈值、四格计数、标签及 B/C 行号。当前已通过。

## 五、D 的待完成工作与推荐步骤

**核心任务 1：五种故障模式的多标签诊断。** 建议以 One-vs-Rest Random Forest 为主线，每个模式独立输出一个连续分数，允许一条记录同时预测多个模式。训练只用 `Group19_A` 的训练集特征和五列 `Y_train`，在验证集选模型配置和各模式阈值；诊断不使用 B/C 的二分类预测作为输入。若要增加其他模型，明确列为扩展，先完成主线。

1. 建立 `Group19_D/`，写配置、训练入口、冻结清单、结果和 README。使用 A 的固定划分、编码和特征；建议从原始输入开始，工程输入作为明确标注的对照。
2. 在验证集**总故障标签为 0**的记录上，为每个模式独立选阈值，目标假阳性率不超过 5%。这是 A 定义的“正常样本”口径；其中可能有与模式标签冲突的记录，需保留并披露。明确阈值并列、是否允许 `score >= threshold`、有限样本下 5% 不能恰好实现时的处理。不要在测试集重新定阈值。
3. 冻结后报告测试集每模式 AP、Precision、Recall、F1、TP/FP/TN/FN、支持数，以及在总故障为 0 的测试记录上实现的 FPR；报告 macro-F1 和 micro-F1，注明零样本或未定义指标的处理。请同时说明“模式正例”与“总故障正例”并非完全一致。
4. 绘制带支持数的模式性能热力图，并给出典型检出、误报、漏报案例。RNF 的测试支持数只有 2，避免据此给出稳定的排名结论。

**核心任务 2：传感器噪声鲁棒性。** A 的配置指定噪声倍率 `0, 0.1, 0.2, 0.5`，随机种子 `101–105`，每个原始数值传感器的尺度只取**训练集标准差（ddof=0）**。对持出集的五个原始数值输入加入高斯噪声；`Type` 不扰动；再用 A 的 `features` 重新计算工程特征。每次使用同一组已拟合预处理器、模型及冻结阈值，不重新训练或重新选阈值。分别报告各倍率跨五次重复的均值与波动，并把零噪声结果与原先预测逐行核对。可先对 D 诊断模型做主实验，再选一组按验证结果预先指定的 B/C 二分类模型作补充，不要根据测试噪声结果选择展示对象。

**可选深化：** 对标签冲突记录做单独切片，说明其对 FPR 与模式召回的影响；按故障模式展示二分类模型的检出覆盖，但应保留各模式支持数，不把重叠模式相加；为关键指标给出固定随机种子 bootstrap 区间，并明确 2 个 RNF 正例的区间不宜作强结论。这些扩展不替代上面两项核心任务。

## 六、建议交付格式与验收清单

结果至少交付 `Group19_D/README.md`、配置、可运行入口、冻结模型/阈值清单、逐行测试预测、每模式指标、噪声重复明细与汇总、模式热力图、鲁棒性图，以及可直接并入最终报告的中英文结论。逐行结果建议保留 `partition, row_index, model, feature_set`，并为每个模式提供真实标签、连续分数、阈值和 0/1 报警；`row_index` 必须对应 A 的原始 CSV 零基数据行位置。B/C 的二分类表使用 `partition, candidate, row_index, model, feature_set, y_true, score, threshold, alarm`，可按 `row_index` 合表。读取浮点 CSV 时可用 `float_precision="round_trip"`。

- [ ] 训练/验证/测试行数 7,000/1,500/1,500；每组行号唯一，五个模式标签与 A 的行号逐行一致。
- [ ] 输入列不含 `UDI`、`Product ID`、总故障或任一模式标签；预处理仅在训练集拟合。
- [ ] 配置、模型、阈值、随机种子和评价口径在测试前冻结并可复现；测试数据未参与选参。
- [ ] 每个模式报告自身正例支持数；测试总故障正常 1,449 行的 FPR 分母清楚；标签冲突没有静默删除。
- [ ] 噪声的标准差来自训练集，五次重复种子固定，工程特征由扰动后的原始数值重新构造。
- [ ] 报告区分基准数据结论与真实部署结论，并对低支持模式给出限制说明。

需要核对完整方法与原始结果时，优先阅读 `Group19_A/HANDOFF.md`、`Group19_B/BC_HANDOFF.md`、`Group19_C/README.md` 和 `Group19_C/C_REPORT.md`。正式对外报告请从各模块 CSV 读取完整精度数据，不要把本交接文件的三位小数当作计算输入。
