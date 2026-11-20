# Group 19：A 模块交接说明

交接日期：2026-11-20。交接对象：B（异常检测）、C（监督分类）、D（诊断与鲁棒性）。

## 1. 当前状态

A 已完成数据审计、固定划分、公共预处理、特征构造、配置文件和 A 模块复现检查。B/C/D 可据此开始接入。模型训练、最终性能评价和全项目端到端复现尚未完成，不能将本包的检查通过理解为所有实验已完成。

已运行验证：四种预处理组合、从原始数据重新生成相同索引、重复运行结果一致性，以及配置、索引、原始文件修改后的拒绝行为。详见 [检查结果](检查结果.md)。

## 2. 仓库放置与首次运行

建议将本目录的**全部内容**放到仓库根目录；已有仓库也可以整体放入 `Group19_A/`，以下命令须先进入该目录。不要只上传代码而漏掉数据、索引及清单。不要把 ZIP 当作仓库源码提交。

Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python run_a.py
.venv\Scripts\python check_a.py
```

macOS / Linux（命令供接入参考，当前验证环境为 Windows / Python 3.12.4）：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_a.py
.venv/bin/python check_a.py
```

建议使用 Python 3.12。两条运行命令应退出码为 0 并输出 PASS。`run_a.py` 会重生成审计结果、复用已冻结索引；`check_a.py` 在临时副本中验证，不改正式数据。依赖首次安装需要网络，数据已包含在包内。

## 3. 必须保留的文件

| 路径 | 用途 |
|---|---|
| `data/raw/ai4i2020.csv` | 原始数据，保持内容及字节不变 |
| `data/splits/indices.npz` | 三组正式划分索引 |
| `data/splits/manifest.json` | 原始数据、索引 SHA-256 与划分配置 |
| `data/splits/train.csv`、`validation.csv`、`test.csv` | 可读索引和 UDI 对照 |
| `pipeline.py` | B/C/D 公共读取、特征和预处理接口 |
| `config.json` | 公共协议与随机种子 |
| `requirements.txt` | 已验证的核心依赖版本 |
| `run_a.py`、`check_a.py` | A 模块运行及回归检查 |
| `results/` | 实际审计输出及参考环境 |

`README.md`、本文件、报告草稿、流程图和检查清单也一并提交。`.gitignore` 排除缓存、虚拟环境与临时文件；不要忽略整个 `data/` 或 `results/`。`.gitattributes` 防止 Git 自动转换原始 CSV 的换行符而破坏其指纹。若目标仓库已有这两个配置文件，请合并规则。

## 4. 接口与最小接入代码

在 `pipeline.py` 同级目录编写或运行接入脚本：

```python
from pipeline import load_split, make_preprocessor

X_train, y_train, Y_train = load_split('train', engineered=True)
X_val, y_val, Y_val = load_split('validation', engineered=True)
pre = make_preprocessor(engineered=True, scale=True)
Z_train = pre.fit_transform(X_train)
Z_val = pre.transform(X_val)
# 用 Z_train 训练模型，用 Z_val 选择模型配置与阈值。
# 二分类目标：y_train；多标签目标：Y_train。
```

- `X` 为 pandas DataFrame；`y` 为 Series；`Y` 为五列 DataFrame，顺序固定为 TWF、HDF、PWF、OSF、RNF。
- 返回对象保留原始行索引且顺序对齐；按位置取行请用 `.iloc`，不要误以为行标签是 0 到 n−1。
- `engineered=False` 使用六项原始输入；`True` 额外添加三个交互特征。读取接口与预处理器的 engineered 参数必须相同。
- Type 编码后，原始组共 8 列，工程组共 11 列。列名可用 `pre.get_feature_names_out()` 获取。
- `scale=True` 仅标准化数值列；树模型可设为 False。各模型保存自己的已拟合预处理器。
- 交叉验证时须在每个训练折内拟合预处理，不得先对整个数据集拟合后再交叉验证。

冻结模型与阈值并记录团队决定后，才执行：

```python
X_test, y_test, Y_test = load_split('test', engineered=True, allow_test=True)
Z_test = pre.transform(X_test)
```

注意：`allow_test=True` 是显式使用确认，不是权限系统。`config.json` 中 `test_model_evaluation_authorized` 当前只是状态记录，代码不会自动用它授权或拒绝访问。接入者仍须遵守团队冻结流程，不能用测试集调参。

## 5. 共同实验协议

1. 固定种子 42，按 Machine failure 分层；训练/验证/测试比例 70/15/15。种子和噪声重复次数是本实现选择，团队应统一采用或在实验开始前共同调整并重新冻结。
2. 不把 UDI、Product ID、Machine failure 或五个故障目标加入模型输入。
3. 不各自重新划分数据，不自行修改或删除标签不一致记录。
4. 模型参数与阈值在验证集确定；AP 为二分类主指标，辅助报告 ROC-AUC、Precision、Recall、F1。
5. 诊断阈值以验证集正常样本 5% 假阳性率为目标，报告测试实际 FPR、每种模式召回率与支持数。正常样本定义为 Machine failure=0；标签冲突带来的影响须说明。逐标签 AP/F1、macro/micro-F1 的阈值规则与未定义指标处理由 D 明确记录。
6. 工程特征为温差、扭矩×转速×2π/60、刀具磨损×扭矩；C 的消融保持划分和模型设置一致。
7. 噪声倍率为 0、0.1、0.2、0.5，标准差仅取训练集（ddof=0）；种子 101–105。在原始数值输入上加噪，再重算工程特征，保持已拟合预处理器和阈值不变，Type 不加噪。

## 6. 数据审计提醒

无缺失和重复输入记录。总故障 339/10,000（3.39%）。27 条标签不一致记录已保存至 `results/label_inconsistencies.csv`：9 条总故障为 1 但所有模式为 0，18 条总故障为 0 但存在模式正例。原始标签保持不变。

| 集合 | 总数 | 总故障 | TWF | HDF | PWF | OSF | RNF |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 7000 | 237 | 33 | 73 | 72 | 71 | 13 |
| validation | 1500 | 51 | 5 | 24 | 10 | 12 | 4 |
| test | 1500 | 51 | 8 | 18 | 13 | 15 | 2 |

五种模式可重叠，不应相加解释为总故障数。测试集 RNF 仅 2 条正例，召回率非常不稳定。以上测试统计属于提案要求的数据审计，不代表已使用测试数据评价模型。

## 7. 各成员下一步及交回内容

| 成员 | 下一步 | 交回 A 的内容 |
|---|---|---|
| B | Isolation Forest、novelty-mode LOF；验证集调参与阈值选择 | 运行脚本、完整参数、分数方向与阈值规则、PR/ROC 输出、方法文字 |
| C | Logistic Regression、Random Forest、XGBoost；普通/加权训练及特征消融 | 脚本、相同调参预算说明、模型对比、特征分析、依赖补充 |
| D | One-vs-Rest 诊断、固定 FPR 评价、噪声实验 | 脚本、逐标签结果与支持数、噪声重复结果、均值波动、图像及错误分析 |
| A | 接入模块，核对配置、数据和结果口径 | 全流程运行命令、最终复现记录与报告/海报一致性检查 |

每位成员提交：可运行入口与命令、依赖变化、配置与种子、结果表/图片、已拟合模型和预处理器的保存/重建说明，以及对应报告文字。建议在各自的 `results/b/`、`results/c/`、`results/d/` 中存放结果，不覆盖 A 的审计表。

建议结果同时记录 model、feature_set、scale、class_weight、seed、partition、threshold、score_direction、metrics、support、数据/索引指纹及 Git commit。没有实际运行的结果不要填入示例数值。

## 8. 提交与验收

- [ ] 将本包源码与数据加入 GitHub，保留数据来源及 AIGC 披露。
- [ ] B/C/D 拉取后运行两条检查命令，在 PR 中记录执行结果。
- [ ] 每人使用独立分支提交，至少一位其他成员审查；A 也须审查一个其他模块。
- [ ] 协议变更经全组讨论，在 PR 写明原因和受影响实验。
- [ ] 模型与阈值冻结后再运行测试；不要根据测试结果返回调参。
- [ ] A 从原始数据复现全部模型表格和图片，再完成最终报告、海报和试讲。

本文件与检查通过不替代上述未完成验收。数据来源、CC BY 4.0 署名和 AI 辅助使用说明见 [README](README.md)；该数据许可不自动等同于项目代码许可证，代码许可由团队决定。
