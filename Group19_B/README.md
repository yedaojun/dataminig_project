# 工作流 B · 异常检测

组员可直接查看[最终测试与 B/C 交接说明](BC_HANDOFF.md)和[误差分析报告](误差分析报告.md)。完整存档包含两轮验证、冻结模型、融合参考分布及最终测试的逐行结果。第二轮验证的候选模型文件（134.46 MiB）使用 Git LFS。需要复现候选冻结时，请在安装 Git LFS 后从仓库根目录运行 `git lfs pull`。第五步最终测试和第六步误差分析可直接使用仓库中的冻结模型和预测。

## 从仓库根目录运行

工作目录为含有 `Group19_A` 和 `Group19_B` 的 `dataminig_project`。本次核验使用 Windows、Python 3.12.7 和 [A 的依赖版本](../Group19_A/requirements.txt)。激活项目虚拟环境后运行下列命令。新环境可先执行 `python -m pip install -r Group19_A/requirements.txt`。

```shell
python -m Group19_B.run validate
```

当前项目已有上一级目录的 Windows 虚拟环境，也可以直接执行 `..\.venv\Scripts\python.exe -B -m Group19_B.run validate`。

| 命令 | 当前行为 |
|---|---|
| `python -m Group19_B.run validate` | 核对开发划分、预处理和行对齐；拟合全部候选，保存验证分数、拟合对象和运行记录。 |
| `python -m Group19_B.run freeze` | 从第二轮候选表定位已拟合模型，选参、校准阈值、验证重载并发布冻结记录。 |
| `python -m Group19_B.run evaluate-test` | 核验冻结记录并重放五组验证分数与报警，再评估冻结测试划分，导出逐行结果、指标及 PR/ROC 图。 |

第五步可直接从仓库根目录运行 `python -m Group19_B.run evaluate-test`。需要已有的冻结模型与融合折外参考，以及 Matplotlib（本次使用 3.11.1，可安装在项目虚拟环境或其基础解释器中）。结果写入 `results/final_test/<清单 SHA-256>/`。只查看结果或与 C 对齐时可直接使用[交接说明和产物索引](BC_HANDOFF.md)。

第六步的[误差分析报告](误差分析报告.md)包含两种基线的配对分歧、可回查案例和融合扩展的故障覆盖情况。在仓库根目录运行 `python -m Group19_B.error_analysis` 可从已验收的第五步结果重建三张分析表。

## 核对开发输入

`validate` 按 [B 的实验配置](config.json) 核对 A 的划分清单，通过 A 的 `load_split` 读取训练和验证划分。每个模型和特征组使用独立的预处理器。训练集调用 `fit_transform`，验证集调用同一预处理器的 `transform`。两种模型都对产品类型作独热编码，LOF 另对数值列作标准化。

| 预处理后的输入 | 训练矩阵 | 验证矩阵 |
|---|---|---|
| 原始组 | 7,000 行 × 8 列 | 1,500 行 × 8 列 |
| 工程组 | 7,000 行 × 11 列 | 1,500 行 × 11 列 |

成功运行以 `PASS` 结束。[开发集行索引](results/development_indices.csv) 含 `split` 和 `row_index` 两列，按训练、验证及各自的冻结顺序保存。`row_index` 是原始 CSV 数据行从零开始的位置。开发入口会核对输入列白名单、禁用列、矩阵有限性，以及特征和二分类标签的索引顺序。

## 候选评分与复核

从仓库根目录执行上面的 `validate` 命令，环境需有 Git 以记录当前提交号。每个候选用全部 7,000 条训练记录的特征拟合，按验证集的冻结顺序计算 `-model.score_samples(X)`。高分表示更异常。两种模型共用行对齐检查和 AP、ROC-AUC 计算函数。统一报警函数按 `score >= threshold` 判定，第 4 环节为每个入选模型校准阈值。

命令最后打印本次产物目录 `results/validation/<清单的 SHA-256>/`。每个目录包含以下三个文件。

| 文件 | 内容与读法 |
|---|---|
| `scores.csv` | 每行对应一条验证记录。前两列为 `row_index` 和原始 `Machine failure`，其余每列对应一个候选 ID。首轮共 16 个候选，第二轮扩参后的完整网格共 32 个。用 pandas 读取时指定 `float_precision="round_trip"` 可还原保存前的浮点分数。 |
| `manifest.json` | 按候选 ID 查模型、特征组、配置顺序、完整参数、数值缩放、输入列顺序及验证 AP、ROC-AUC。所有候选共用的种子、数据及索引指纹见 `baseline`。另记录环境、Git 基线、实际源码及两个产物的 SHA-256。 |
| `fits.joblib` | `models` 以候选 ID 保存各配置的原始拟合对象，`preprocessors` 以清单中的 `preprocessor_key` 保存 4 个训练集拟合的预处理器，供后续选参和冻结时复用。 |

## 重现阈值校准

在仓库根目录运行 `python -m Group19_B.run freeze`。命令从 [第二轮候选表](results/candidates_round2.csv) 的 `run_id` 找到对应的验证产物，重新计算 32 组验证 AP 和 ROC-AUC。每个模型和特征组按完整精度 AP 选参，平局取网格中靠前者。对各入选配置，按分数从高到低逐组纳入同分记录，用整数比值比较故障类 F1，平局取较高阈值。

成功后，`results/frozen/CURRENT` 指向已核验的冻结目录。`manifest.json` 记录每个模型的完整阈值、特征顺序、依赖版本、文件指纹和判定规则。`models.joblib` 保存四组已拟合模型及其预处理器。`metrics.csv` 是验证集汇总表，`predictions.csv` 包含原始行索引、候选 ID、真实标签、分数、阈值和报警。复算请以 `float_precision="round_trip"` 读取 CSV，按完整精度执行 `score >= threshold`。
