# Group 19 — A 部分交付包

**团队交接入口：[HANDOFF.md](HANDOFF.md)**。包含 GitHub 上传范围、B/C/D 接入方法、共同协议与验收清单。

已完成数据审计、固定划分、统一特征与预处理接口，以及 A 模块复现验证。B/C/D 模型尚未接入，因此全项目端到端复现待后续完成。随机种子 42 和噪声种子为本实现选择，供团队统一采用。

## 快速运行

在本目录打开终端（建议 Python 3.12）：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python run_a.py
.venv\Scripts\python check_a.py
```

原始数据已经包含在 `data/raw/ai4i2020.csv`，运行不需要下载。`results/` 为实际运行产物。CSV 是机器可读审计输出，不需要用 Excel 手动修改。

## 给 B/C/D 的接入示例

```python
from pipeline import load_split, make_preprocessor
X_train, y_train, Y_train = load_split('train', engineered=True)
X_val, y_val, Y_val = load_split('validation', engineered=True)
pre = make_preprocessor(engineered=True, scale=True)
X_train_processed = pre.fit_transform(X_train)
X_val_processed = pre.transform(X_val)
# B/C 使用 y_train；D 使用 Y_train，列顺序为 TWF,HDF,PWF,OSF,RNF。
# 冻结配置和阈值后，才允许：
# X_test, y_test, Y_test = load_split('test', engineered=True, allow_test=True)
# X_test_processed = pre.transform(X_test)
```

`engineered=False` 为原始特征；`True` 为原始特征加三个交互特征。one-hot 后分别为 8 列与 11 列。`scale=True` 标准化数值列，Type 编码不标准化；树模型可使用 `scale=False`。每个模型保留自己的已拟合预处理器。不能在验证集或测试集重新调用 `fit`；若进行交叉验证，也必须在每个训练折内拟合预处理。

## 固定数据与配置约定

- `data/splits/indices.npz` 是正式划分；CSV 同时给出零起始行号与 UDI，UDI 只用于追溯，不参与训练。
- `manifest.json` 记录原始 CSV 的 SHA-256、种子与比例。重复运行复用索引；原始数据或划分配置变化会中止运行。
- 标签不一致记录不删除、不重标。主任务始终使用原始 Machine failure，多标签任务使用原始五列标签。
- 原始数据没有重复输入记录；本实现没有进行样本删除、重采样或缺失填补。
- `config.json` 定义共同协议；模型参数由 B/C/D 各自记录。建议结果记录 model、seed、features、class_weight、validation threshold、metrics、版本信息。
- 噪声实验使用训练集数值传感器标准差（ddof=0），只扰动 held-out 原始数值输入，再重新构造工程特征；Type 不扰动。保持预处理器与阈值不变，由 D 实现实验。
- 测试集标签仅用于计划要求的数据审计和计数；本交付包没有训练模型或评价测试性能。

## 文件说明

- `run_a.py`：审计、划分冻结、预处理验证及结果生成。
- `check_a.py`：在临时副本中从原始数据重建划分，并检查配置、索引、原始数据被修改时能否及时拒绝。
- `pipeline.py`：团队共用接口。
- `results/label_inconsistencies.csv`：27 条不一致记录，含原始 UDI。
- `results/split_counts.csv`：各集合总量与故障模式数量。
- `results/mode_overlap.csv`：故障两两重叠计数，对角线为各标签总数。
- `results/mode_combinations.csv`：完整标签组合频数。
- `A部分报告草稿.md`：可用于报告的中文方法、结果、限制及交付状态。
- `pipeline.svg`：可用于海报和报告的流程示意图。
- `复现检查清单.md`：已完成和待集成项目。

修订后的划分清单同时记录索引文件 SHA-256。运行入口和数据读取接口均检查数据、索引与划分配置的一致性，检查失败不会重写已有审计输出。已发布索引保持不变。

## 数据来源与许可

AI4I 2020 Predictive Maintenance Dataset (2020), UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C5HS5C
页面：https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
下载：https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip
获取日期：2026-11-19。数据采用 CC BY 4.0，署名见上述来源。本数据为合成基准。
审计以下载的 CSV 为准，不能用网页叙述中的故障数量替换实际计数。

## 协作与披露

本包由 AI 辅助生成代码、分析和报告草稿，并已在当前环境实际运行。提交前负责人应阅读、理解和审查实现，按课程规定如实披露 AIGC 使用。共享仓库、团队 PR 审查、其他成员模块接入及最终海报答辩需要团队后续完成。
