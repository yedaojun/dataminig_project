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

工程输入提高了测试宏 AP 和 F1，但提升并不均匀。`HDF`、`PWF`、`OSF`
的工程输入测试 AP 均约为 1.000，且召回率均为 100%；`TWF` 反而由 raw 的 AP
0.166 降至 0.071。固定 5% 正常样本误报预算意味着五种模式各自都保留了一批报警，
因此单模式精确率仍低：工程输入宏精确率仅 0.134。五模式报警取并集后，raw 和
engineered 分别在 266 / 1,449 与 228 / 1,449 条测试正常记录上至少报警一次，
对应 18.36% 与 15.74% 的 any-mode FPR。逐模式 5% 预算不能解释为整机 5% 误报。

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

作为预先指定的二分类补充，B 的 raw LOF 是按验证 AP 选出的最高 raw 模型。其测试 AP
从零噪声的 0.223 ± 0.000 降至 0.5 倍噪声的 0.104 ± 0.014，F1 从
0.276 ± 0.000 降至 0.129 ± 0.013。与此同时，实测正常样本 FPR 从 3.38% 升至
26.64%。0.5 倍噪声下召回看似上升，主要来自报警膨胀，不能解读为鲁棒性改善。

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
python -m Group19_D.check_results
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
- Frozen protocol and hashes: [freeze_manifest.json](results/freeze_manifest.json)

Read floating-point CSV files with `float_precision="round_trip"` when checking
threshold boundaries.
