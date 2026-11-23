# Workstream C: supervised binary failure detection

This module uses A's frozen 70/15/15 split and feature constructor. It fits logistic regression, random forest, and XGBoost on the 7,000 training rows, selects hyperparameters and alarm thresholds using only the 1,500 validation rows, then evaluates the 1,500 test rows. The five failure-mode labels, UDI, Product ID, and `Machine failure` are never predictors.

## Reproduce from the repository root

Use a project environment. Install `Group19_A/requirements.txt` and then `Group19_C/requirements.txt`. The recorded run used Python 3.13.5, NumPy 2.3.1, pandas 3.0.0, scikit-learn 1.7.1, XGBoost 3.4.1, and Matplotlib 3.10.5; the A requirements pin earlier versions and recommend Python 3.12. To avoid unpickling across incompatible library versions, run `tune` afresh after installing dependencies.

```shell
python -m Group19_C.run tune
python -m Group19_C.analyze_ablation
python -m Group19_C.run importance
python -m Group19_C.run evaluate-test
python -m Group19_C.check_results
python -m Group19_C.plot_results
python -m Group19_C.extended_analysis
python -m Group19_C.check_extended
```

Run `tune` and inspect the frozen validation results **before** `evaluate-test`. The latter checks hashes of the data, split, config, code, and serialized models before opening the test split. It never refits. Rerunning `tune` creates a new freeze and can overwrite the previous freeze; do not use test results to revise the grid or select a model.

## Experimental protocol

Each model has three prespecified parameter candidates for each feature set and weighting option: 3 × 2 × 2 × 3 = 36 fits. All groups get the same three-candidate tuning budget. The candidate with the highest validation average precision (AP) wins; an exact tie keeps the first candidate in config order. Each selected model gets its own threshold from the distinct observed validation scores, maximizing positive-class F1; a tie uses the highest threshold. Alarms are `score >= threshold`. The score is the estimated positive-class probability. The test results include every validation-selected family × feature set × weighting group; no group is chosen after viewing test results.

Logistic regression standardizes numeric inputs and one-hot encodes `Type`. Tree models use unscaled numeric inputs and the same encoding. Weighted logistic regression and forest use `class_weight="balanced"`; weighted XGBoost uses `scale_pos_weight = train negatives / train positives = 28.5358649789`. All preprocessing is fitted on the training split. Forests use 150 trees and `min_samples_leaf=2`; XGBoost uses 200 boosting rounds, learning rate 0.05, and histogram tree construction. Configured parameter grids are in [config.json](config.json).

## Outputs

| File | Contents |
|---|---|
| `results/candidates.csv` | All 36 validation scores and exact parameters |
| `results/matched_feature_ablation.csv`, `matched_weight_ablation.csv` | Matched candidate contrasts on validation AP, holding model settings fixed |
| `results/validation_metrics.csv`, `validation_predictions.csv` | The 12 selected validation models, thresholds, and row-level scores |
| `results/freeze_manifest.json`, `frozen_models.joblib` | Frozen protocol, fingerprints, fitted preprocessors and models |
| `results/test_metrics.csv`, `test_predictions.csv` | One-time test metrics and row-level scores for all 12 models |
| `results/bc_comparison.csv` | B's raw and engineered baselines beside C's corresponding models |
| `results/permutation_importance.csv` | Three-repeat validation AP decrease after permuting each original input column, for the higher-validation-AP weighting per family and feature set |
| `results/model_comparison.png`, `feature_importance.png`, `pr_roc.png` | Presentation figures, rebuilt with `plot_results.py` |
| `results/paired_bootstrap.csv` | 2,000 paired, class-stratified test-row bootstrap replicates summarized as AP and AP-difference percentile intervals |
| `results/alarm_budgets.csv`, `alarm_budgets.png` | Binary failure recall and realized false-positive rate at 1%, 2%, and 5% budgets calibrated only on validation normals |
| `results/label_efficiency_repeats.csv`, `label_efficiency_summary.csv`, `label_efficiency.png` | Validation AP after refitting the frozen settings on 10%, 25%, 50%, and 100% of labeled training rows |

`test_predictions.csv` follows B's `partition, candidate, row_index, model, feature_set, y_true, score, threshold, alarm` contract. It adds no test-tuned decisions. Read floating-point CSV with `float_precision="round_trip"` when checking exact threshold boundaries. `check_results.py` recomputes AP and ROC-AUC and checks labels, counts, alarms, unique row IDs, and alignment with B.

## Interpretation

See [C_REPORT.md](C_REPORT.md) for the findings and report-ready text. The substantial engineered-feature gain for forest and XGBoost reflects known relationships used to generate this synthetic benchmark. Permutation importance is descriptive and can understate correlated inputs; it is not a causal attribution. The fixed 51 positive test rows also limit precision of comparisons. Noise robustness and per-mode diagnosis belong to D's workstream.

## Additional evidence protocol

`extended_analysis.py` reads the already-frozen C models and B/C predictions. It selects the examples in its comparison panel **only by saved validation AP**: the best B raw detector, best C raw classifier, and best C engineered classifier. For the feature comparison it also includes the unweighted raw and engineered forests; these share identical hyperparameters. No test metric chooses the panel. The script verifies C's data, split, code, config, and model fingerprints before use.

The bootstrap samples the same test row IDs for both models, drawing failure and normal rows separately with replacement to keep the observed 51/1,449 class ratio. The 2.5th and 97.5th percentiles describe uncertainty from these **fixed-model, fixed-test-set row resamples**. They do not include model-training variability or establish real-world generalization. The 1%, 2%, and 5% alarm limits are calibrated on the 1,449 validation records with `Machine failure == 0`, with ties handled by setting the threshold just above the boundary score. A target validation FPR is not guaranteed on test. These binary alarm budgets are distinct from D's per-mode diagnostic thresholds.

For label efficiency, the chosen C model hyperparameters stay fixed while 10%, 25%, and 50% of the labeled training **rows** are sampled within each class for five seeds (201–205); the 100% point fits all training rows once. Each subset gets a newly fitted preprocessor and model. AP is measured only on the existing validation split. The same validation split originally selected the full-data hyperparameters, so this is a descriptive sensitivity analysis, not an independent model-selection study or a semi-supervised experiment. `check_extended.py` recomputes the key counts and confirms the full-data AP matches the original validation results.
