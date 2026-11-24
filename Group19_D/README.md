# Workstream D: five-mode diagnosis and sensor-noise robustness

This module completes Group 19's diagnostic workstream. It uses A's frozen
70/15/15 split and public preprocessing contract, trains an independent
one-vs-rest random forest for each of `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`,
freezes validation-selected models and per-mode thresholds, and only then
evaluates the 1,500-row test split. It also repeats the frozen pipeline under
Gaussian sensor noise.

## Reproduce from the repository root

Install `Group19_A/requirements.txt`, then `Group19_D/requirements.txt`. The
recorded run uses Python 3.12.7, NumPy 1.26.4, pandas 2.2.2, scikit-learn
1.5.1, Matplotlib 3.9.2, and joblib 1.4.2.

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

Run `tune` and inspect `results/validation_metrics.csv` before
`evaluate-test`. `evaluate-test` verifies the data, split, configuration,
source, and serialized-model hashes before reading test. It never refits or
retunes. The noise script performs a second hash check and reuses every frozen
preprocessor, model, and threshold.

The corruption extension loads Group19_C's frozen models, including XGBoost.
Create and run it in a separate environment so the core project environment is
not changed:

```shell
conda env create -f Group19_D/environment_corruption.yml
conda run -n aiaa3111_corruption python -B -m Group19_D.corruption_analysis
conda run -n aiaa3111_corruption python -B -m Group19_D.label_conflict_sensitivity
conda run -n aiaa3111_corruption python -B -m Group19_D.leave_one_type_out
conda run -n aiaa3111_corruption python -B -m Group19_D.repeated_splits
```

## Diagnostic protocol

For each input set, three random-forest configurations are fitted separately
for all five modes. The configuration with the highest mean validation AP
across the five modes is selected; an exact tie keeps the first configuration
in `config.json`. This grid is fixed before test evaluation.

| Input set | Columns | Numeric scaling |
|---|---|---|
| `raw` | five sensor readings plus one-hot product type | none |
| `engineered` | raw inputs plus A's temperature difference, mechanical power, and wear-load features | none |

Each mode receives its own threshold using only validation rows with
`Machine failure == 0`. With 1,449 such rows, 5% permits at most 72 alarms. The
threshold is placed immediately above the 73rd-highest normal score, and an
alarm is `score >= threshold`. This handles ties and the finite-sample budget
explicitly. Validation-normal rows with an inconsistent mode label remain in
the denominator and are not removed.

The test report therefore separates two negative-class definitions:

- `normal_fpr`: alarms divided by the 1,449 test rows with total failure 0.
- `mode_negative_fpr`: alarms divided by rows where that specific mode is 0.

Macro metrics average the five mode-specific values. Micro metrics pool
true/false positives and negatives over all 7,500 test mode rows. A record may
be positive for multiple modes, so the five supports must not be summed to
describe the number of failed machines.

## Noise protocol

Independent Gaussian noise is added to the five raw numeric sensor values at
multipliers `0`, `0.1`, `0.2`, and `0.5` of A's training standard deviations
(`ddof=0`). Seeds are `101` through `105`; `Type` is unchanged. Engineered
features are recomputed from the perturbed raw readings. No model,
preprocessor, or threshold is changed.

At zero noise, every diagnostic score is checked row by row against the frozen
test predictions. The supplementary binary comparison is B's raw LOF model,
selected because it has the highest validation AP among B's raw-input models.
That selection is based on validation results alone.

## Outputs

| File | Contents |
|---|---|
| `results/candidate_mode_metrics.csv`, `candidates.csv` | Validation AP/AUC for every candidate and mode, plus the selected candidate per input set |
| `results/validation_predictions.csv`, `validation_metrics.csv` | Frozen validation scores, thresholds, mode metrics, and normal/mode-negative FPR denominators |
| `results/frozen_models.joblib`, `freeze_manifest.json` | Fitted preprocessors and all mode models, hashes, selected parameters, and exact thresholds |
| `results/test_predictions.csv`, `test_metrics.csv` | Row-level test scores and alarms; per-mode AP, ROC-AUC, precision, recall, F1, confusion counts, support, and both FPR definitions |
| `results/test_aggregates.csv` | Macro and micro test metrics plus the any-mode alarm rate on total-failure normals |
| `results/mode_support.csv` | Train/validation/test support for every mode and total failure |
| `results/cases.csv` | Highest-scoring detected, false-positive, and missed examples with raw input context |
| `results/mode_performance_heatmap.png` | Test AP and F1 by mode/input set with support labels |
| `results/mode_pr_curves.png` | Test precision-recall curves for all ten mode/input-set combinations |
| `results/noise_repeats.csv`, `noise_summary.csv` | Per-mode noise repeats and their mean/SD/min/max summaries |
| `results/noise_aggregate_repeats.csv`, `noise_aggregate_summary.csv` | Macro AP/F1 and any-mode normal FPR across the five repeats |
| `results/noise_robustness.png`, `noise_fpr.png` | D AP/F1 and realized normal-FPR robustness |
| `results/binary_noise_repeats.csv`, `binary_noise_summary.csv`, `binary_noise_robustness.png` | B raw LOF robustness supplement |
| `results/bootstrap_replicates.csv`, `bootstrap_intervals.csv` | 2,000 stratified test-row bootstrap replicates and 95% percentile intervals |
| `results/macro_ap_sensitivity.csv` | Macro AP including/excluding RNF and support-weighted sensitivity summaries |
| `results/threshold_anchors.csv`, `threshold_calibration.csv` | Rank-budget, empirical-quantile, and fixed-0.05 threshold comparison |
| `results/threshold_system_summary.csv` | Macro/micro F1 and any-mode FPR under each threshold method |
| `results/bootstrap_intervals.png`, `threshold_calibration.png` | Presentation figures for uncertainty and threshold sensitivity |
| `Group19_D/oracle_rules.json`, `results/oracle_metrics.csv` | Fixed AI4I generator rules and rule-oracle metrics on all partitions |
| `results/oracle_rule_checks.csv` | TP/FP/FN and mismatch counts over all 10,000 rows and each split |
| `results/oracle_vs_d.csv`, `oracle_vs_d.png` | Oracle-vs-model test comparison with stochastic/unavailable modes identified |
| `results/permutation_importance_by_mode.csv` | Validation-only five-repeat permutation AP drop for every mode and input feature |
| `results/mode_feature_importance_heatmap.png` | Per-mode feature-importance heatmaps for raw and engineered inputs |
| `results/corruption_mode_repeats.csv`, `corruption_mode_summary.csv` | D per-mode results under clean, Gaussian, +1-SD bias, and 10% missing corruption |
| `results/corruption_binary_repeats.csv`, `corruption_binary_summary.csv` | Frozen C engineered RF and B raw LOF binary comparisons under every corruption |
| `results/corruption_d_aggregate_repeats.csv`, `corruption_d_aggregate_summary.csv` | D macro AP/F1 and any-mode FPR by corruption condition |
| `results/corruption_*_comparison.png`, `corruption_mode_heatmap.png` | Presentation figures for binary and five-mode corruption sensitivity |
| `results/label_conflict_rows.csv` | The 27 audited conflicts by split, row ID, total label, and five mode labels |
| `results/label_conflict_d_mode_metrics.csv`, `label_conflict_d_aggregate.csv` | D diagnostic sensitivity under retain, drop, and mode-derived total labels |
| `results/label_conflict_binary_metrics.csv` | C binary-model sensitivity under the same three strategies |
| `results/label_conflict_*_predictions.csv` | Row-level predictions and alarms for every strategy |
| `results/label_conflict_*_sensitivity.png` | Presentation figures for label-conflict sensitivity |
| `results/alarm_operating_points.csv` | Frozen C/B/D thresholds expressed as recall and normal FPR |
| `results/alarm_cost_scenarios.csv` | Expected alarms and costs per 1,000 devices across prevalence and cost-ratio scenarios |
| `results/alarm_cost_actual_prevalence.csv` | Per-1,000-device scenario at the observed 3.4% test failure prevalence |
| `results/alarm_cost_per_1000.png`, `alarm_cost_sensitivity.png` | Operational alarm/cost comparisons |
| `results/loto_d_mode_metrics.csv`, `loto_d_aggregate.csv` | D per-mode and macro results when each product Type is held out |
| `results/loto_binary_metrics.csv` | C binary results for the same leave-one-type-out folds |
| `results/loto_support.csv` | Train/validation/test rows, failures, and mode supports by held-out Type |
| `results/loto_*_predictions.csv` | Row-level LOTO scores and alarms for D and C |
| `results/loto_d_comparison.png`, `loto_c_comparison.png` | In-domain vs leave-one-type-out comparisons |
| `results/repeated_split_d_aggregate.csv`, `repeated_split_d_mode_metrics.csv` | D results for model refits under split seeds 42–46 |
| `results/repeated_split_binary_metrics.csv` | C engineered RF results under the same five splits |
| `results/repeated_split_summary.csv` | Mean, sample SD, min, and max by model/feature set |
| `results/repeated_split_*_stability.png` | Split-to-split stability figures |
| `results/system_fpr_search.csv` | Full validation search over common per-mode allowances for system union-FPR budgets |
| `results/system_thresholds.csv` | Selected per-mode thresholds for 1%, 2%, and 5% union-FPR targets |
| `results/system_alarm_metrics.csv` | Validation/test union FPR, union recall, precision, and macro F1 |
| `results/system_alarm_cost.csv` | Per-1,000-device cost at a 10:1 missed:false ratio |
| `results/system_alarm_calibration.png` | System-level FPR and recall tradeoff |

`test_predictions.csv` uses:

```text
partition, row_index, candidate, model, feature_set, mode,
y_true, total_failure, score, threshold, alarm
```

`row_index` is the zero-based row position in A's unchanged raw CSV. Read
floating-point CSV files with `float_precision="round_trip"` when checking
threshold boundaries.

## Interpretation limits

This is diagnosis on a synthetic predictive-maintenance benchmark, not
remaining-useful-life prediction or evidence of production performance. The
same total-failure label generates several mode labels, and 27 audited label
conflicts are retained. Test support is especially small for `RNF` (two
positive records), so its AP, recall, and noise variation cannot justify a
stable ranking. Engineered features mirror generator relationships and their
gain should not be generalized to real machinery without external validation.
