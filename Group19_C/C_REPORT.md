# Workstream C results and report text

## Findings (中文摘要)

所有模型都使用 A 的固定划分；测试集包含 51 个故障和 1,449 个正常样本。以下 AP、F1 均为测试集结果，但候选配置和 F1 阈值在验证集确定。`raw` 是与 B 的主要对照，`engineered` 额外加入温差、机械功率和磨损负载。

| 模型 | 输入 | 训练 | 验证 AP | 测试 AP | 测试 F1 | TP | FP |
|---|---|---|---:|---:|---:|---:|---:|
| Logistic regression | raw | 普通 | 0.385 | 0.425 | 0.429 | 18 | 15 |
| Logistic regression | raw | 加权 | 0.385 | 0.385 | 0.387 | 18 | 24 |
| Random forest | raw | 普通 | 0.732 | 0.756 | 0.690 | 39 | 23 |
| Random forest | raw | 加权 | 0.642 | 0.750 | 0.655 | 36 | 23 |
| XGBoost | raw | 普通 | 0.775 | 0.802 | 0.759 | 41 | 16 |
| XGBoost | raw | 加权 | 0.690 | 0.807 | 0.758 | 36 | 8 |
| Logistic regression | engineered | 普通 | 0.391 | 0.475 | 0.449 | 22 | 25 |
| Logistic regression | engineered | 加权 | 0.393 | 0.454 | 0.431 | 22 | 29 |
| Random forest | engineered | 普通 | 0.913 | 0.917 | 0.882 | 41 | 1 |
| Random forest | engineered | 加权 | 0.879 | 0.902 | 0.835 | 43 | 9 |
| XGBoost | engineered | 普通 | 0.862 | 0.849 | 0.851 | 43 | 7 |
| XGBoost | engineered | 加权 | 0.847 | 0.865 | 0.863 | 41 | 3 |

工程特征使普通随机森林测试 AP 从 0.756 升到 0.917，普通 XGBoost 从 0.802 升到 0.849；逻辑回归仅从 0.425 升到 0.475。为了隔离选参影响，另按完全相同模型参数配对比较验证 AP：普通随机森林的三个配置平均提升 0.203，普通 XGBoost 平均提升 0.100，普通逻辑回归平均提升 0.007。验证集上的森林置换重要性排名中，转速、温差、磨损负载均重要。相关特征可能分摊重要性，故不能把排序解释为单个传感器的因果贡献。

类别加权并非一致获益：相同参数配对的验证 AP 在两组树模型里均下降；普通和加权 Logistic 的原始组验证 AP 接近。测试集加权 XGBoost 的 AP 个别高于普通配置，属于已冻结方案的事后观察，不能据此重新选参。B 的原始孤立森林和 LOF 测试 AP 分别为 0.164 和 0.223，低于 C 的各原始监督模型；但 B 的方法不使用故障标签拟合，监督与无监督的训练信息不同。B 的工程组测试 AP 为 0.335 和 0.325，另有融合扩展 AP 0.581，不应把融合当成原始输入主对照。

## Report-ready English section

We evaluated three supervised model families under the shared frozen split: logistic regression, random forest, and XGBoost. For each family, we compared unweighted and class-weighted training on raw and engineered inputs. Each of the twelve family–feature–weighting groups had three prespecified hyperparameter candidates. We selected the highest validation average precision (AP) within each group and selected its alarm threshold by maximizing validation positive-class F1, breaking ties toward a higher threshold. The final test set was accessed only after these choices and the fitted models were frozen.

On raw inputs, the unweighted logistic regression, random forest, and XGBoost achieved test APs of 0.425, 0.756, and 0.802, respectively. Their test F1 scores at the frozen thresholds were 0.429, 0.690, and 0.759. The corresponding B anomaly detectors achieved test APs of 0.164 (Isolation Forest) and 0.223 (LOF). This comparison favors the supervised models on this benchmark, while reflecting their use of labeled failure examples during training.

Engineered inputs particularly helped the nonlinear tree models: unweighted random forest reached test AP 0.917 and F1 0.882, while unweighted XGBoost reached AP 0.849 and F1 0.851. Holding hyperparameters fixed across feature sets, the three unweighted forest candidates improved validation AP by 0.203 on average, and the three unweighted XGBoost candidates improved by 0.100. Validation-only permutation analysis for the engineered forest highlighted rotational speed, temperature difference, and wear-load interaction. These interaction features encode relationships related to the synthetic data generator, so the gain cannot be read as evidence of transfer to real machinery. Class weighting did not consistently improve validation AP or test F1. The small number of positive test records (51) and correlated predictors limit the certainty and interpretation of detailed rankings.

## Figures and data

- [Test AP comparison](results/model_comparison.png)
- [Engineered forest permutation importance](results/feature_importance.png)
- [Test PR and ROC curves](results/pr_roc.png)
- Exact metrics: [validation](results/validation_metrics.csv), [test](results/test_metrics.csv), [B/C comparison](results/bc_comparison.csv)
- Row-level test predictions: [test_predictions.csv](results/test_predictions.csv)

## Additional evidence: uncertainty, alarm load, and training sample size

The exploratory comparison panel was defined from saved **validation AP**, without selecting models by the additional test summaries: raw B LOF, raw C unweighted XGBoost, and engineered C unweighted random forest. The raw and engineered unweighted forests also provide a matched-parameter feature comparison. A paired bootstrap resampled test row IDs 2,000 times, separately within the 51 failures and 1,449 normals. The resulting test AP difference for raw C XGBoost minus raw B LOF was **0.579**, with a 95% percentile interval of **0.443–0.685**. For engineered minus raw unweighted random forest, the AP difference was **0.161** (interval **0.086–0.243**). These intervals quantify row-sampling variability for fixed fitted models and this benchmark; they exclude training and dataset-shift uncertainty. See [paired_bootstrap.csv](results/paired_bootstrap.csv).

We also calibrated binary alarm thresholds on validation normals to limit the validation false-positive rate to at most 1%, 2%, or 5%. At the **1% target**, engineered random forest detected **45/51** test failures with **7/1,449** false alarms; raw XGBoost detected **39/51** with **14/1,449**, and raw B LOF detected **6/51** with **11/1,449**. At the 5% target, engineered forest detected 48 failures but produced 76 false alarms. This makes the recall–alarm tradeoff explicit. The 5% target led to **5.25% realized test FPR** for the engineered forest, illustrating that validation calibration does not guarantee the same rate on new records. These thresholds are an additional operating-point analysis; they do not replace the previously frozen F1 thresholds. See [alarm_budgets.csv](results/alarm_budgets.csv) and [alarm_budgets.png](results/alarm_budgets.png).

Finally, with model settings held fixed, we refitted the best raw and engineered C models on stratified subsets of labeled training rows (five repeats per subset). Using 10% of the training rows, about 24 positive examples, the engineered forest averaged validation AP **0.762 ± 0.066**, and raw XGBoost averaged **0.475 ± 0.116**; with all 7,000 rows they reached **0.913** and **0.775**. At 50% of training rows, the corresponding means were **0.898 ± 0.015** and **0.723 ± 0.037**. The error bars are standard deviations across five subset seeds, not confidence intervals. This analysis suggests performance depends on access to labeled failures, particularly for raw XGBoost. It uses the same validation set that selected the original full-data hyperparameters and should be treated as descriptive. See [label_efficiency_summary.csv](results/label_efficiency_summary.csv) and [label_efficiency.png](results/label_efficiency.png).

### Report-ready English addendum

To probe the reliability and operational meaning of the supervised comparison, we performed three exploratory analyses without changing the frozen candidate choices. A paired, class-stratified bootstrap of the test rows estimated an AP advantage of 0.579 (95% percentile interval 0.443–0.685) for the validation-selected raw XGBoost over the validation-selected raw LOF. With identical forest hyperparameters, engineered inputs improved test AP by 0.161 (interval 0.086–0.243) relative to raw inputs. At a threshold calibrated to at most 1% false positives among validation normals, the engineered forest detected 45 of 51 test failures with seven false alarms among 1,449 test normals. A training-size sensitivity experiment, refitting unchanged model settings on stratified labeled subsets, showed that the engineered forest's mean validation AP rose from 0.762 with 10% of training rows to 0.913 with all rows. Bootstrap intervals describe variation within this fixed test set, and the training-size experiment uses the original validation set; neither analysis establishes deployment performance.
