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

`test_predictions.csv` follows B's `partition, candidate, row_index, model, feature_set, y_true, score, threshold, alarm` contract. It adds no test-tuned decisions. Read floating-point CSV with `float_precision="round_trip"` when checking exact threshold boundaries. `check_results.py` recomputes AP and ROC-AUC and checks labels, counts, alarms, unique row IDs, and alignment with B.

## Interpretation

See [C_REPORT.md](C_REPORT.md) for the findings and report-ready text. The substantial engineered-feature gain for forest and XGBoost reflects known relationships used to generate this synthetic benchmark. Permutation importance is descriptive and can understate correlated inputs; it is not a causal attribution. The fixed 51 positive test rows also limit precision of comparisons. Noise robustness and per-mode diagnosis belong to D's workstream.
