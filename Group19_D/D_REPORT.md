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
python -m Group19_D.check_results
```

The corruption extension uses a separate Conda environment:

```shell
conda env create -f Group19_D/environment_corruption.yml
conda run -n aiaa3111_corruption python -B -m Group19_D.corruption_analysis
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
  [oracle_vs_d.csv](results/oracle_vs_d.csv)
- Per-mode feature importance:
  [permutation_importance_by_mode.csv](results/permutation_importance_by_mode.csv)
- Corruption comparison:
  [corruption_binary_summary.csv](results/corruption_binary_summary.csv),
  [corruption_d_aggregate_summary.csv](results/corruption_d_aggregate_summary.csv)
- Frozen protocol and hashes: [freeze_manifest.json](results/freeze_manifest.json)

Read floating-point CSV files with `float_precision="round_trip"` when checking
threshold boundaries.
