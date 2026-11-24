# Workstream D results and report text

## Findings (中文摘要)

D 工作流使用 A 冻结的 7,000 / 1,500 / 1,500 划分，以 One-vs-Rest
Random Forest 独立诊断 `TWF`、`HDF`、`PWF`、`OSF`、`RNF`。每个输入组先在验证集上
按五模式宏平均 AP 选参，再在 1,449 条 `Machine failure == 0` 的验证记录上为每个
模式独立校准阈值。5% 目标允许每个阈值最多产生
`floor(0.05 × 1,449) = 72` 个验证正常样本报警；阈值取在预算内最后一个允许分数之上，
报警规则为 `score >= threshold`。模型、预处理器和十组模式阈值均在读取测试集前冻结。

| 输入组 | 选中配置 | 验证宏 AP | 测试宏 AP | 测试宏 F1 | 测试 micro-F1 |
|---|---|---:|---:|---:|---:|
| raw | 深度不限，叶节点最少 2 | 0.5006 | 0.5425 | 0.1904 | 0.2004 |
| engineered | 最大深度 12，叶节点最少 1 | 0.6173 | 0.6234 | 0.2267 | 0.2420 |

工程输入提高了含 RNF 的五模式宏 AP 和 F1，但这一口径受极稀疏模式影响。
排除 RNF 后，raw 与 engineered 的宏 AP 分别为 0.674 和 0.768；按测试支持数加权的 AP
分别为 0.741 和 0.862。`HDF`、`PWF`、`OSF`
的工程输入测试 AP 均约为 1.000，且召回率均为 100%；`TWF` 反而由 raw 的 AP
0.166 降至 0.071。固定 5% 正常样本误报预算意味着五种模式各自都保留了一批报警，
因此单模式精确率仍低：工程输入宏精确率仅 0.134。五模式报警取并集后，raw 和
engineered 分别在 266 / 1,449 与 228 / 1,449 条测试正常记录上至少报警一次，
对应 18.36% 与 15.74% 的 any-mode FPR。逐模式 5% 预算不能解释为整机 5% 误报。

对测试行按 `Machine failure` 分层重采样 2,000 次后，含 RNF 的宏 AP 区间为
raw `0.543 [0.478, 0.621]`、engineered `0.623 [0.608, 0.677]`；排除 RNF 后为
`0.674 [0.593, 0.770]` 与 `0.768 [0.757, 0.809]`。支持加权、排除 RNF 的 AP
分别为 `0.741 [0.644, 0.851]` 与 `0.862 [0.786, 0.947]`。含 RNF 的宏 AP
只有 1,698 / 2,000 次重采样可定义，因为部分重采样没有抽到 RNF 正例。因此主结论
应采用排除 RNF 的宏 AP 或支持加权 AP，RNF 本身只作描述性报告。

Any-mode FPR 的分层 bootstrap 区间为 raw `18.36% [16.36%, 20.43%]`、
engineered `15.74% [13.80%, 17.60%]`。区间只描述固定模型和固定测试集上的行抽样
波动，不包含重新训练、分布漂移或真实部署不确定性。

| 模式 | 测试支持 | raw AP | raw F1 | raw FPR | engineered AP | engineered F1 | engineered FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| TWF | 8 | 0.166 | 0.110 | 5.73% | 0.071 | 0.103 | 6.42% |
| HDF | 18 | 0.762 | 0.303 | 5.04% | 1.000 | 0.371 | 3.73% |
| PWF | 13 | 0.794 | 0.226 | 4.62% | 1.000 | 0.289 | 3.73% |
| OSF | 15 | 0.975 | 0.288 | 4.21% | 1.000 | 0.323 | 3.59% |
| RNF | 2 | 0.015 | 0.025 | 5.11% | 0.046 | 0.048 | 2.62% |

表中的 FPR 分母均为 1,449 条测试总故障正常记录。验证集 5% 目标不保证测试集也
不超过 5%；工程输入的 `TWF` 即达到 6.42%。`RNF` 只有 2 个测试正例，两种输入
均检出 1 个、漏报 1 个；其 AP、F1 和噪声波动不能支持稳定的模式排名。

噪声倍率取训练标准差 `0 / 0.1 / 0.2 / 0.5`（`ddof=0`），重复种子为
`101–105`。噪声只加入五个原始数值传感器，`Type` 保持不变，工程特征由扰动后的
原始读数重新计算。五模式宏平均 AP 的均值如下：

| 噪声倍率 | raw 宏 AP | engineered 宏 AP | raw 宏 F1 | engineered 宏 F1 | raw 并集 FPR | engineered 并集 FPR |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.543 | 0.623 | 0.190 | 0.227 | 18.36% | 15.74% |
| 0.1 | 0.490 | 0.514 | 0.163 | 0.187 | 21.28% | 20.11% |
| 0.2 | 0.401 | 0.395 | 0.137 | 0.135 | 25.42% | 27.09% |
| 0.5 | 0.217 | 0.190 | 0.075 | 0.065 | 40.91% | 49.10% |

零噪声的每次重复都与冻结测试预测逐行一致。工程特征在低噪声下仍有优势，但在
0.2 倍噪声时宏 AP 已略低于 raw，在 0.5 倍噪声下两类输入的宏 AP 和 F1 都严重下降，
五模式报警并集 FPR 也升至 raw 40.91% 与 engineered 49.10%。
这说明本数据上工程特征的收益对传感器扰动敏感，不能仅依据零噪声结果宣称其鲁棒。

阈值分析区分了三种方法。当前 `rank_budget` 规则按“允许 72 个验证正常报警”定位
排序边界；OSF 的 `4.94e-324` 与 PWF 的 `8.09e-05` 是随机森林离散概率在该排序边界
上的数值表现，不代表概率校准良好。经验分位数加 tie-adjusted 规则在多数模式上与
当前数值阈值相同，系统并集 FPR 仅从 engineered 15.735% 变为 15.666%。固定
概率阈值 0.05 在测试集上报警更少（engineered 并集 FPR 4.90%），但它并未按验证
集 5% 预算校准，不能与当前方法直接视为同一操作点。

## 生成器规则 oracle 与特征重要性

为避免把漂亮的 AP 误解为真实设备上的模型能力，我们使用 AI4I 文档中的生成规则
构造了固定、不可拟合的 oracle baseline：

| 模式 | 固定规则 | 规则性质 | 测试 AP | 测试 F1 | 标签完全复原 |
|---|---|---|---:|---:|---|
| TWF | 200 ≤ 磨损 ≤ 240 | 随机暴露窗口 | 0.034 | 0.083 | 否 |
| HDF | 温度差 < 8.6 K 且转速 < 1380 rpm | 确定性 | 1.000 | 1.000 | 是 |
| PWF | 机械功率 < 3500 W 或 > 9000 W | 确定性 | 1.000 | 1.000 | 是 |
| OSF | 磨损 × 扭矩 > Type 阈值 `L/M/H = 11000/12000/13000` | 确定性 | 1.000 | 1.000 | 是 |
| RNF | 无特征确定性规则 | 随机注入 | 不报告 | 不报告 | 否 |

HDF、PWF、OSF 的规则在训练、验证、测试集上都 100% 复原原始模式标签，说明
逐行核对表进一步确认：这三个确定性规则在全部 10,000 条记录以及 train /
validation / test 三个划分上均为 `FP = 0, FN = 0, mismatches = 0`。TWF 窗口规则
在全量数据有 750 条不匹配，RNF 的 19 条正例全部无法由规则检出。这个结果说明
learned RF 的 AP≈1.000 与 rule baseline 是两种不同证据：前者是排序表现，
后者是逐行确定性核对；engineered RF 在 HDF/PWF/OSF 的操作点 F1 仍为
0.29–0.37，并不等于规则 baseline 的 F1=1。

engineered 输入接近完美的 AP 主要是生成器关系恢复。TWF 的窗口只是随机故障的
暴露条件；RNF 完全随机。因此不能把 five-mode headline 写成“模型发现了五个真实
故障规律”。把四种可用规则直接取 OR 后，总故障测试 AP/F1 为 0.288 / 0.456，
进一步说明总标签中仍包含模型或随机组件无法由确定性规则复原的部分。

验证集五模式 permutation importance（每特征重复 5 次）与规则结构一致：

- HDF：raw 输入首要特征是空气温度和转速；engineered 输入首要特征是转速
  （AP 下降 0.629）和温差（0.601）。
- PWF：engineered 输入首要特征是机械功率，AP 下降 0.759。
- OSF：engineered 输入首要特征是磨损负载，AP 下降 0.706，其次为 Type。
- TWF：首要特征仍是工具磨损；engineered 输入中磨损负载也提供次一级信息。
- RNF：验证集只有 4 个正例，importance 数值只作描述，不用于特征排名。

这些结果支持“生成器逆向”解释，而不是模型可迁移性的证据。

作为预先指定的二分类补充，B 的 raw LOF 是按验证 AP 选出的最高 raw 模型。其测试 AP
从零噪声的 0.223 ± 0.000 降至 0.5 倍噪声的 0.104 ± 0.014，F1 从
0.276 ± 0.000 降至 0.129 ± 0.013。与此同时，实测正常样本 FPR 从 3.38% 升至
26.64%。0.5 倍噪声下召回看似上升，主要来自报警膨胀，不能解读为鲁棒性改善。

## 非 iid 传感器腐蚀与 C 模型补充

为检查 iid Gaussian 之外的失效模式，我们在同一冻结模型和阈值上增加：

- 五个原始数值传感器同时加训练集标准差的固定 `+1 SD` 偏置；
- 每个传感器独立随机缺失 10% 单元格，并使用训练集中位数恢复；
- 缺失值恢复后重新计算 D 的温差、机械功率和磨损负载；
- C 选择验证 AP 最高的 `engineered_RandomForest_unweighted_1`，阈值固定为
  `0.494642`；B 继续使用预先选定的 raw LOF。模型和阈值均未重训或重调。

固定偏置比缺失和 iid Gaussian 更致命。D 的 raw / engineered 五模式宏 AP
分别降至 `0.148 / 0.123`，五模式并集 FPR 升至 `72.60% / 86.40%`。C 的测试 AP
从 clean 的 `0.917` 降至 `0.127`，FPR 从 `0.07%` 升至 `49.62%`。B raw LOF 的
排序 AP 虽升至 `0.260`，但召回率变为 100% 的同时 FPR 达到 `91.72%`，F1 仅
`0.071`，属于报警膨胀而非鲁棒性提升。

10% 随机缺失并恢复中位数的影响相对温和：D raw / engineered 宏 AP 为
`0.449 / 0.481`，并集 FPR 为 `18.80% / 20.32%`；C AP/F1 为
`0.726 / 0.707`，FPR 为 `0.95%`。这说明固定偏置造成的阈值漂移远比随机缺失的
方差影响严重。Engineered 特征在缺失条件下仍略优于 raw，但在偏置条件下不再占优，
其零噪声优势不能解释为稳健性。

## 27 条标签冲突敏感性

三个口径分别为：

1. `retain`：保留原始 `Machine failure`，即当前主结果；
2. `drop_conflicts`：从 train / validation / test 删除 27 条冲突记录，并用原选中
   超参数重新拟合 D 和 C；
3. `total_from_modes`：把总故障定义为五种模式的 OR，D 模式模型不变，但按新二分类
   正常样本重新校准阈值。

27 条冲突的分布为 train 19、validation 6、test 2；其中 9 条总故障为 1 但五模式
全 0，18 条总故障为 0 但至少一个模式为 1。删除测试冲突后 RNF 支持数由 2 降为 0，
因此跨口径只能比较排除 RNF 的四模式宏 AP。

| 口径 | C AP | C F1 | C FPR | D raw 四模式宏 AP | D eng. 四模式宏 AP | raw union FPR | eng. union FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| retain | 0.917 | 0.882 | 0.07% | 0.674 | 0.768 | 18.36% | 15.74% |
| drop | 0.914 | 0.915 | 0.00% | 0.674 | 0.773 | 15.41% | 13.61% |
| modes define total | 0.897 | 0.860 | 0.00% | 0.674 | 0.768 | 18.38% | 15.69% |

结论对冲突处理基本稳定。C 的 AP 最大下降为 0.020，D 的四模式宏 AP 变化不超过
0.005；删除冲突后并集 FPR 略低，部分原因是删除了 2 条总故障为 0、但某模式为 1
的测试记录。该下降不能解释为模型更容易部署。

## 每 1,000 台设备的报警成本

为避免 FPR/F1 停留在抽象指标，我们把冻结阈值换算为每 1,000 台设备、每个检查
周期的期望计数。故障率使用测试集 `51 / 1,500 = 3.4%`，并另外对
1% / 5% / 10% 故障率做敏感性分析。设一次误报审核成本为 1 单位，一次漏检成本为
`R` 单位：

```text
TP = 1000 × prevalence × recall
FP = 1000 × (1 - prevalence) × normal_FPR
FN = 1000 × prevalence × (1 - recall)
cost = FP + R × FN
```

在 3.4% 故障率和 `R = 10` 时：

| 操作点 | TP | FP | FN | 总报警 | 成本单位 |
|---|---:|---:|---:|---:|---:|
| C engineered RF，F1 阈值 | 27.3 | 0.7 | 6.7 | 28.0 | 67.3 |
| C engineered RF，1% 验证 FPR | 30.0 | 4.7 | 4.0 | 34.7 | 44.7 |
| C raw XGB，F1 阈值 | 27.3 | 10.7 | 6.7 | 38.0 | 77.3 |
| B raw LOF，F1 阈值 | 10.7 | 32.7 | 23.3 | 43.3 | 266.0 |
| D raw 五模式并集 | 32.0 | 177.3 | 2.0 | 209.3 | 197.3 |
| D engineered 五模式并集 | 33.3 | 152.0 | 0.7 | 185.3 | 158.7 |

成本比从 1:1 增加到 50:1 时，C engineered RF 的 F1 点成本从 7.3 增至 334.0，
1% 预算点从 8.7 增至 204.7；D engineered 并集仅从 152.7 增至 185.3，因为其成本
早已被大量误报主导。该换算清楚说明：五模式诊断每模式 5% 不能直接用于整机报警，
需要先优化系统级并集预算。这里的“每台设备”假设每条记录对应一次设备检查，若一台
设备有多个检查周期，应乘以周期数，并考虑同一设备报警的相关性。

## Leave-one-product-type-out

跨产品类型实验以每种 Type 为一次留出：

- 在 A 的 train 中删除该 Type 后拟合；
- 在 A 的 validation 中删除该 Type 后校准阈值；
- 只在 A 的 test 中属于该 Type 的行上评估；
- 另将原先在全部 Type 上训练的冻结模型在同一测试切片上作为 in-domain reference。

由于 `Type` 的一热编码只见过训练折中的类别，留出 Type 在测试时会编码为全零；
因此这比随机划分更严格，模拟了产品类型本身未被模型见过的情况。

| 留出 Type | 测试行/故障 | C AP reference→LOTO | C F1 reference→LOTO | D eng. 宏 AP reference→LOTO | D eng. union FPR |
|---|---:|---:|---:|---:|---:|
| L | 885 / 26 | 0.927→0.707 | 0.939→0.622 | 0.635→0.544 | 15.83%→15.95% |
| M | 472 / 20 | 0.900→0.855 | 0.919→0.865 | 0.780→0.765 | 12.83%→16.59% |
| H | 143 / 5 | 0.810→0.967 | 0.286→0.800 | 1.000→0.625 | 24.64%→13.04% |

留出 L 时性能下降最大，表明 Type 分布变化会影响 C 和 D。留出 M 时结果较稳定。
H 只有 5 个总故障，且测试中 PWF/OSF/RNF 均为 0 正例，D 只能评估 TWF/HDF 两个
模式；其看似有利的 AP/FPR 变化不能作为稳定结论。总体而言，随机划分结果不能直接
解释为跨产品类型泛化，尤其应以 L 的结果作为主要风险证据。

## 五次重复分层划分

以 seed `42–46` 分别重建 70/15/15 分层划分，保持 D/C 的原选中超参数，在每个
划分中重新拟合、只在验证集校准阈值，再评估测试集。Seed 42 与 A 的冻结训练索引
逐行一致。五次结果使用样本标准差（`ddof=1`），表示划分敏感性，不是置信区间。

| 模型/输入 | 重复数 | 均值 ± 标准差 | 最小–最大 |
|---|---:|---:|---:|
| D raw 宏 AP | 5 | 0.557 ± 0.020 | 0.528–0.573 |
| D engineered 宏 AP | 5 | 0.611 ± 0.007 | 0.607–0.623 |
| D raw union FPR | 5 | 18.98% ± 1.12% | 17.74%–20.36% |
| D engineered union FPR | 5 | 16.78% ± 1.29% | 15.74%–18.22% |
| C engineered RF AP | 5 | 0.894 ± 0.041 | 0.835–0.932 |
| C engineered RF F1 | 5 | 0.872 ± 0.047 | 0.796–0.915 |

Engineered D 在五个种子上均优于 raw 宏 AP，平均差约 0.055；其并集 FPR 也均低于
raw。C 的 AP 波动较大，seed 45 为最低值 0.835。总体而言，五模式诊断的优势方向
稳定，但单一划分下的具体小数和 C 的绝对 AP 不应被过度精确解读。

## 系统级并集 FPR 校准

原主结果的 5% 是“每个模式各自 5%”，不能直接作为整机报警预算。系统级版本在验证
正常样本上搜索一个共同 per-mode 允许误报数 `k`，要求五模式并集不超过目标 1% /
2% / 5%，再冻结阈值评估测试集。该策略仍使用原 D scores，不重新训练或改选模型。

| 输入 | 验证目标 | per-mode `k` | 测试 TP | 测试 FP | 测试 union FPR | union recall | 成本/1000 台 |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1% | 2 | 27 | 12 | 0.83% | 52.94% | 168.0 |
| raw | 2% | 5 | 38 | 29 | 2.00% | 74.51% | 106.0 |
| raw | 5% | 16 | 43 | 71 | 4.90% | 84.31% | 100.7 |
| engineered | 1% | 2 | 44 | 11 | 0.76% | 86.27% | 54.0 |
| engineered | 2% | 6 | 45 | 37 | 2.55% | 88.24% | 64.7 |
| engineered | 5% | 16 | 45 | 73 | 5.04% | 88.24% | 88.7 |

与原来的 engineered“每模式 5%”相比，系统级 1% 预算把测试误报从 228 降到 11，
同时因 five-mode union 仍可检出大部分故障，成本在 10:1 口径下从 158.7 降到 54.0。
系统级 5% 与原始设置召回相同，但误报从 228 降到 73。验证目标不保证测试值完全一致：
engineered 2% 和 5% 的测试 union FPR 分别为 2.55% 和 5.04%。因此报告应同时给出
验证目标与测试实现值，不能把验证预算当作部署保证。

## Report-ready English section

We completed the five-mode diagnostic workstream with independent one-vs-rest
random forests trained on A's fixed 7,000-row training split. Three
prespecified forest configurations were compared for each input set by mean
validation average precision across `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`.
Each mode then received its own threshold on the 1,449 validation records with
total failure 0. The 5% target allowed at most 72 false alarms per mode, and
the exact threshold was placed above the highest excluded normal score before
the test split was accessed. Test evaluation did not refit or retune.

On raw inputs, the five-mode test macro AP was 0.543 and macro F1 at the frozen
thresholds was 0.190. Engineered inputs increased macro AP to 0.623 and macro
F1 to 0.227. The gain was concentrated in `HDF`, `PWF`, and `OSF`, for which
engineered-input AP and recall reached approximately 1.000 and 100%. In
contrast, `TWF` AP fell from 0.166 on raw inputs to 0.071 on engineered
inputs. Because each of five modes received its own 5% normal-FPR allowance,
alarms from different modes formed a union with an 18.36% raw-input and 15.74%
engineered-input false-positive rate on the 1,449 test normals. A union-level
false-positive rate above the per-mode target is expected when several
independent indicators are monitored.

A 2,000-replicate row bootstrap stratified by total-failure status quantified
this uncertainty. Including RNF, macro AP was 0.543 [0.478, 0.621] for raw
inputs and 0.623 [0.608, 0.677] for engineered inputs. Excluding the two-case
RNF mode increased these estimates to 0.674 [0.593, 0.770] and 0.768
[0.757, 0.809]. Support-weighted AP excluding RNF was 0.741 [0.644, 0.851]
and 0.862 [0.786, 0.947], respectively. The all-five-mode estimate was
undefined in 302 of 2,000 engineered-input replicates because no RNF positive
was sampled. RNF is therefore reported for completeness but excluded from
headline ranking claims. The any-mode normal-FPR interval was 18.36%
[16.36%, 20.43%] for raw inputs and 15.74% [13.80%, 17.60%] for engineered
inputs.

The engineered features recover relationships used by the synthetic generator;
the AP increase is not evidence of generalization to real machinery. This
interpretation is especially important for HDF, PWF, and OSF, for which the
engineered features align closely with the benchmark's deterministic rules.

Under Gaussian sensor noise scaled by training-set standard deviations, the
engineered-input macro AP declined from 0.623 at zero noise to 0.514, 0.395,
and 0.190 at multipliers 0.1, 0.2, and 0.5. Raw macro AP declined from 0.543
to 0.490, 0.401, and 0.217. Any-mode normal FPR rose from 15.74% to 49.10%
for engineered inputs and from 18.36% to 40.91% for raw inputs over the same
range. Thus engineered features helped in the noiseless
benchmark, but their advantage disappeared at the 0.2 multiplier and reversed
at 0.5. Zero-noise scores matched the original test predictions row by row.
The supplementary raw LOF detector had test AP 0.223 with 3.38% normal FPR at
zero noise, falling to AP 0.104 while normal FPR increased to 26.64% at the
0.5 multiplier. The increased recall at high noise therefore reflects alarm
inflation rather than improved detection.

The current diagnostic thresholds are rank-budget thresholds, not calibrated
probabilities. The operating point is defined by excluding rank 73 among the
1,449 validation normals; consequently, near-zero numerical thresholds for
PWF and OSF reflect discrete random-forest scores at that rank boundary. An
empirical-quantile threshold with the same finite-sample tie handling produced
nearly identical system behavior: engineered any-mode FPR changed only from
15.735% to 15.666%. A fixed probability threshold of 0.05 reduced test alarms
but is a different, stricter operating point and was not selected using the 5%
validation-normal budget.

To separate model behavior from generator recovery, we evaluated fixed rules
from the AI4I generator documentation without fitting any parameters. HDF,
PWF, and OSF rules reproduced their mode labels exactly on all three
partitions and achieved test AP and F1 of 1.000. The TWF exposure rule,
200–240 minutes of tool wear, is only a stochastic sampling window and reached
AP/F1 0.034/0.083. RNF has no deterministic feature rule and is not assigned
an oracle AP. The OR of the four available rules reached total-failure AP/F1
0.288/0.456, demonstrating that deterministic generator components do not
explain the complete failure label.

Validation-only permutation importance aligns with these formulas. For
engineered inputs, HDF depends primarily on rotational speed and temperature
difference, PWF on mechanical power, and OSF on the wear-load interaction.
The corresponding AP drops were 0.629/0.601 for HDF, 0.759 for PWF, and 0.706
for OSF across five repeats. TWF remains centered on tool wear; RNF has only
four validation positives and is not ranked. These results support the
interpretation that engineered inputs recover generator relationships rather
than demonstrate deployment model capability.

We extended robustness testing beyond iid Gaussian noise. A fixed +1-SD bias
was applied to all five numeric sensors, and a second condition independently
removed 10% of numeric-sensor cells and restored them with training medians
before recomputing engineered features. The validation-best C model
(`engineered_RandomForest_unweighted_1`) and the preselected B raw LOF model
were evaluated under the same frozen thresholds.

The fixed bias was substantially more damaging than missingness. D's raw and
engineered five-mode macro AP fell to 0.148 and 0.123, with any-mode normal FPR
rising to 72.60% and 86.40%. C AP fell from 0.917 to 0.127 while its FPR rose
to 49.62%. B's ranking AP increased to 0.260, but recall reached 100% only
through a 91.72% normal-FPR increase, leaving F1 at 0.071. Under 10% random
missingness, D macro AP was 0.449 raw and 0.481 engineered, with union FPR
18.80% and 20.32%; C reached AP/F1 0.726/0.707 at 0.95% FPR. This indicates
that threshold drift from constant bias is more consequential than the
additional variance from this missing-data protocol.

We also evaluated three treatments of the 27 audited total/mode-label
conflicts: retaining the original target, dropping all conflict rows and
refitting, and defining total failure as the OR of the five mode labels.
There were 19 conflicts in train, 6 in validation, and 2 in test. Dropping the
test conflicts removed both RNF positives, so the cross-strategy diagnostic
comparison uses macro AP over the four interpretable modes.

The four-mode macro AP was effectively unchanged: raw remained 0.674 in all
three strategies; engineered changed from 0.768 to 0.773 under dropping and
0.768 under mode-derived totals. C AP changed from 0.917 to 0.914 under
dropping and 0.897 under mode-derived totals. Union FPR changed from 15.74%
to 13.61% and 15.69% for engineered D. The lower FPR after dropping partly
reflects removal of two test conflict records and is not evidence of improved
deployment performance. The main conclusions are therefore robust to the
conflict treatment, with the RNF support caveat made explicit.

We translated frozen operating points into expected counts per 1,000 device
inspections. At the observed 3.4% failure prevalence and a missed-failure cost
ten times the false-alarm review cost, the C engineered RF F1 threshold
produced 27.3 detections, 0.7 false alarms, 6.7 misses, and 67.3 cost units
per 1,000 devices. Its 1% validation-FPR point produced 30.0 detections, 4.7
false alarms, 4.0 misses, and 44.7 cost units. D's engineered five-mode union
detected 33.3 failures but generated 152.0 false alarms, totaling 185.3 alarms
and 158.7 cost units. Raw LOF reached only 10.7 detections with 32.7 false
alarms and 266.0 cost units.

The qualitative ranking remains stable as the missed:false cost ratio grows
from 1:1 to 50:1: C's 1% operating point is less costly than its F1 point for
ratios above one, while the D union remains dominated by false alarms. These
counts assume one test row equals one inspection. Multiple inspections per
device require multiplication by cycle count and should account for correlated
alarms on the same device.

We also performed leave-one-product-type-out evaluation. For each held-out
Type, models were refit on the other two types in A's training split,
thresholds were calibrated on validation rows from those observed types, and
performance was measured only on test rows of the held-out type. Because the
one-hot encoder sees only observed types, the held-out Type is encoded as
all-zero at test time.

C binary AP changed from 0.927 to 0.707 when L was held out, from 0.900 to
0.855 for M, and from 0.810 to 0.967 for H. Engineered D macro AP changed from
0.635 to 0.544 for L, 0.780 to 0.765 for M, and 1.000 to 0.625 for H. The L
shift is the clearest evidence that the random-split results should not be
treated as type-invariant generalization. The H result is much less reliable:
its test split has five total failures, and PWF, OSF, and RNF have no positive
test examples, so D's H macro AP uses only TWF and HDF.

Finally, we recreated five 70/15/15 stratified splits using seeds 42–46 and
refit the fixed D/C settings in every split. D's raw macro AP averaged
0.557 ± 0.020 and engineered macro AP 0.611 ± 0.007. Engineered inputs had
higher macro AP in all five splits; the average difference was approximately
0.055. Any-mode FPR averaged 18.98% ± 1.12% for raw inputs and
16.78% ± 1.29% for engineered inputs. C engineered RF averaged AP
0.894 ± 0.041 and F1 0.872 ± 0.047, with seed 45 the weakest split at AP
0.835. These are sample standard deviations across five split seeds, not
confidence intervals. They indicate a stable direction for the engineered
diagnostic advantage, but wide enough variation that single-split decimals
and C's absolute AP should not be over-interpreted.

We also replaced five independent 5% per-mode budgets with a system-level
union constraint. A common per-mode validation allowance was selected as the
largest value whose five-mode union remained within 1%, 2%, or 5%; those
thresholds were then frozen for test. For engineered inputs, the 1% validation
budget detected 44 of 51 failures with 11 test false alarms (union FPR 0.76%).
The 5% budget detected 45 failures with 73 false alarms (5.04%). At a 10:1
missed:false cost ratio and 3.4% prevalence, the corresponding expected costs
were 54.0 and 88.7 units per 1,000 devices, compared with 158.7 for the
original independent-5% thresholds. The system-level 1% point therefore
substantially reduced alarm load with a smaller recall loss. Validation
budgets were not exact test guarantees: engineered test FPR was 2.55% for a
2% validation target and 5.04% for a 5% target.

The principal limitation is statistical support. The test split contains only
eight `TWF`, eighteen `HDF`, thirteen `PWF`, fifteen `OSF`, and two `RNF`
positive records. `RNF` conclusions are especially unstable. Five overlapping
mode labels are not five independent failure counts, and 27 audited
total-failure/mode-label conflicts were retained. These results characterize a
synthetic benchmark and should not be interpreted as expected performance for
real machinery or as remaining-useful-life prediction.

## Reproduction and evidence

```shell
python -m Group19_D.run tune
python -m Group19_D.run evaluate-test
python -m Group19_D.noise
python -m Group19_D.extended_analysis
python -m Group19_D.oracle_analysis
python -m Group19_D.alarm_cost_analysis
python -m Group19_D.system_alarm_calibration
python -m Group19_D.check_results
```

The corruption extension uses a separate Conda environment:

```shell
conda env create -f Group19_D/environment_corruption.yml
conda run -n aiaa3111_corruption python -B -m Group19_D.corruption_analysis
conda run -n aiaa3111_corruption python -B -m Group19_D.label_conflict_sensitivity
conda run -n aiaa3111_corruption python -B -m Group19_D.leave_one_type_out
conda run -n aiaa3111_corruption python -B -m Group19_D.repeated_splits
```

- Exact mode metrics: [test_metrics.csv](results/test_metrics.csv),
  [validation_metrics.csv](results/validation_metrics.csv)
- Macro/micro results and union FPR: [test_aggregates.csv](results/test_aggregates.csv)
- Row-level test results: [test_predictions.csv](results/test_predictions.csv)
- Selected examples: [cases.csv](results/cases.csv)
- Noise repeats and summaries: [noise_repeats.csv](results/noise_repeats.csv),
  [noise_summary.csv](results/noise_summary.csv),
  [noise_aggregate_summary.csv](results/noise_aggregate_summary.csv)
- B binary supplement: [binary_noise_summary.csv](results/binary_noise_summary.csv)
- Bootstrap intervals: [bootstrap_intervals.csv](results/bootstrap_intervals.csv)
- RNF and macro-AP sensitivity: [macro_ap_sensitivity.csv](results/macro_ap_sensitivity.csv)
- Threshold comparison: [threshold_calibration.csv](results/threshold_calibration.csv),
  [threshold_system_summary.csv](results/threshold_system_summary.csv)
- Generator-rule oracle: [oracle_metrics.csv](results/oracle_metrics.csv),
  [oracle_rule_checks.csv](results/oracle_rule_checks.csv),
  [oracle_vs_d.csv](results/oracle_vs_d.csv)
- Per-mode feature importance:
  [permutation_importance_by_mode.csv](results/permutation_importance_by_mode.csv)
- Corruption comparison:
  [corruption_binary_summary.csv](results/corruption_binary_summary.csv),
  [corruption_d_aggregate_summary.csv](results/corruption_d_aggregate_summary.csv)
- Label-conflict sensitivity:
  [label_conflict_binary_metrics.csv](results/label_conflict_binary_metrics.csv),
  [label_conflict_d_aggregate.csv](results/label_conflict_d_aggregate.csv)
- Alarm cost per 1,000 devices:
  [alarm_cost_actual_prevalence.csv](results/alarm_cost_actual_prevalence.csv),
  [alarm_operating_points.csv](results/alarm_operating_points.csv)
- Leave-one-type-out:
  [loto_d_aggregate.csv](results/loto_d_aggregate.csv),
  [loto_binary_metrics.csv](results/loto_binary_metrics.csv)
- Repeated splits:
  [repeated_split_summary.csv](results/repeated_split_summary.csv),
  [repeated_split_d_aggregate.csv](results/repeated_split_d_aggregate.csv)
- System-level FPR calibration:
  [system_alarm_metrics.csv](results/system_alarm_metrics.csv),
  [system_thresholds.csv](results/system_thresholds.csv)
- Frozen protocol and hashes: [freeze_manifest.json](results/freeze_manifest.json)

Read floating-point CSV files with `float_precision="round_trip"` when checking
threshold boundaries.
