"""Calibrate D's five-mode thresholds to a system-level union-FPR budget.

For each target union-FPR budget (1%, 2%, 5%), search the largest common
per-mode validation false-alarm allowance k whose union alarm rate is still
within budget. The selected mode thresholds are then frozen and applied to
test. This preserves the raw and engineered score rankings while replacing
five independent 5% budgets with one system-level operating constraint.

Run after D evaluate-test/extended analyses:
    python -m Group19_D.system_alarm_calibration
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Group19_A.pipeline import MODES
from Group19_D.extended_analysis import load_partition
from Group19_D.run import OUT, mode_metrics, read_config, verify_freeze

REPO = Path(__file__).resolve().parent.parent
BUDGETS = (0.01, 0.02, 0.05)
DEVICES = 1000
TEST_PREVALENCE = 51 / 1500
MISS_TO_FALSE_COST_RATIO = 10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def threshold_at_normal_rank(scores, total_failure, allowed_fp) -> float:
    normal = np.asarray(total_failure) == 0
    normal_scores = np.sort(np.asarray(scores)[normal])
    if allowed_fp < 0 or allowed_fp >= len(normal_scores):
        raise ValueError("Invalid finite-sample allowance")
    boundary = normal_scores[len(normal_scores) - allowed_fp - 1]
    return float(np.nextafter(boundary, np.inf))


def alarm_matrix(scores, thresholds):
    return np.column_stack([
        scores[:, index] >= thresholds[index]
        for index in range(len(MODES))
    ])


def search_budget(validation, budget):
    total = validation["total_failure"]
    normal_count = int((total == 0).sum())
    allowed_union_fp = int(np.floor(budget * normal_count))
    normal = total == 0
    search_rows = []
    best = None
    for allowed_per_mode in range(allowed_union_fp + 1):
        thresholds = np.asarray([
            threshold_at_normal_rank(
                validation["scores"][:, mode_index],
                total,
                allowed_per_mode,
            )
            for mode_index in range(len(MODES))
        ])
        alarms = alarm_matrix(validation["scores"], thresholds)
        union_normal_fp = int(alarms[normal].any(axis=1).sum())
        search_rows.append({
            "target_union_fpr": budget,
            "allowed_union_normal_fp": allowed_union_fp,
            "per_mode_allowed_fp": allowed_per_mode,
            "per_mode_fpr": allowed_per_mode / normal_count,
            "validation_union_normal_fp": union_normal_fp,
            "validation_union_normal_fpr": union_normal_fp / normal_count,
            "within_budget": union_normal_fp <= allowed_union_fp,
            "thresholds": thresholds,
        })
        if union_normal_fp <= allowed_union_fp:
            best = {
                "allowed_per_mode_fp": allowed_per_mode,
                "thresholds": thresholds,
                "union_normal_fp": union_normal_fp,
                "union_normal_fpr": union_normal_fp / normal_count,
                "allowed_union_fp": allowed_union_fp,
            }
    if best is None:
        raise AssertionError("Even the strictest threshold exceeded the budget")
    return best, search_rows


def evaluate_partition(data, thresholds):
    total = data["total_failure"]
    labels = data["labels"]
    scores = data["scores"]
    alarms = alarm_matrix(scores, thresholds)
    normal = total == 0
    failure = total == 1
    union_alarm = alarms.any(axis=1)
    tp_union = int(np.sum(union_alarm & failure))
    fp_union = int(np.sum(union_alarm & normal))
    fn_union = int(np.sum(~union_alarm & failure))
    mode_rows = []
    for mode_index, mode in enumerate(MODES):
        mode_rows.append(mode_metrics(
            labels[:, mode_index],
            scores[:, mode_index],
            thresholds[mode_index],
            total,
        ))
    macro_f1 = float(np.mean([row["f1"] for row in mode_rows]))
    macro_recall = float(np.mean([row["recall"] for row in mode_rows]))
    return {
        "union_tp": tp_union,
        "union_fp": fp_union,
        "union_fn": fn_union,
        "union_normal_denominator": int(normal.sum()),
        "union_failure_denominator": int(failure.sum()),
        "union_normal_fpr": fp_union / int(normal.sum()),
        "union_failure_recall": tp_union / int(failure.sum()),
        "union_precision": (
            tp_union / (tp_union + fp_union)
            if tp_union + fp_union
            else np.nan
        ),
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
    }


def cost_scenarios(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in metrics[metrics.partition == "test"].itertuples():
        failures = DEVICES * TEST_PREVALENCE
        normals = DEVICES - failures
        tp = failures * row.union_failure_recall
        fn = failures - tp
        fp = normals * row.union_normal_fpr
        rows.append({
            "feature_set": row.feature_set,
            "target_union_fpr": row.target_union_fpr,
            "devices": DEVICES,
            "test_prevalence": TEST_PREVALENCE,
            "expected_tp": tp,
            "expected_fp": fp,
            "expected_fn": fn,
            "expected_alarms": tp + fp,
            "expected_cost_units": fp + MISS_TO_FALSE_COST_RATIO * fn,
            "miss_to_false_cost_ratio": MISS_TO_FALSE_COST_RATIO,
        })
    return pd.DataFrame(rows)


def figures(metrics: pd.DataFrame):
    test = metrics[metrics.partition == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for feature_set, color, marker in (
        ("raw", "#777777", "o"),
        ("engineered", "#5B0F78", "D"),
    ):
        frame = test[test.feature_set == feature_set].sort_values(
            "target_union_fpr"
        )
        x = frame.target_union_fpr.to_numpy() * 100
        axes[0].plot(
            x,
            frame.union_normal_fpr * 100,
            marker=marker,
            linewidth=2,
            color=color,
            label=feature_set.title(),
        )
        axes[1].plot(
            x,
            frame.union_failure_recall * 100,
            marker=marker,
            linewidth=2,
            color=color,
            label=feature_set.title(),
        )
    axes[0].plot([1, 2, 5], [1, 2, 5], "--", color="black", linewidth=1, label="Target")
    axes[0].set(xlabel="Validation union-FPR target (%)", ylabel="Test union FPR (%)")
    axes[1].set(xlabel="Validation union-FPR target (%)", ylabel="Test failure recall (%)")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.suptitle("D system-level alarm calibration")
    fig.savefig(OUT / "system_alarm_calibration.png", dpi=180)
    plt.close(fig)


def main():
    config = read_config()
    verify_freeze()
    validation_sets = load_partition("validation")
    test_sets = load_partition("test")
    search_rows = []
    threshold_rows = []
    metric_rows = []
    for feature_set in config["feature_sets"]:
        validation = validation_sets[feature_set]
        test = test_sets[feature_set]
        for budget in BUDGETS:
            best, rows = search_budget(validation, budget)
            for row in rows:
                row = row.copy()
                row["feature_set"] = feature_set
                row["thresholds"] = json.dumps(row["thresholds"].tolist())
                search_rows.append(row)
            thresholds = best["thresholds"]
            for mode, threshold in zip(MODES, thresholds):
                threshold_rows.append({
                    "feature_set": feature_set,
                    "target_union_fpr": budget,
                    "per_mode_allowed_fp": best["allowed_per_mode_fp"],
                    "mode": mode,
                    "threshold": threshold,
                })
            for partition, data in (("validation", validation), ("test", test)):
                values = evaluate_partition(data, thresholds)
                metric_rows.append({
                    "feature_set": feature_set,
                    "target_union_fpr": budget,
                    "partition": partition,
                    "per_mode_allowed_fp": best["allowed_per_mode_fp"],
                    "per_mode_fpr": (
                        best["allowed_per_mode_fp"]
                        / values["union_normal_denominator"]
                    ),
                    **values,
                })
    search = pd.DataFrame(search_rows)
    thresholds = pd.DataFrame(threshold_rows)
    metrics = pd.DataFrame(metric_rows)
    costs = cost_scenarios(metrics)
    search.to_csv(
        OUT / "system_fpr_search.csv",
        index=False,
        float_format="%.17g",
    )
    thresholds.to_csv(
        OUT / "system_thresholds.csv",
        index=False,
        float_format="%.17g",
    )
    metrics.to_csv(
        OUT / "system_alarm_metrics.csv",
        index=False,
        float_format="%.17g",
    )
    costs.to_csv(
        OUT / "system_alarm_cost.csv",
        index=False,
        float_format="%.17g",
    )
    figures(metrics)
    manifest = {
        "budgets": list(BUDGETS),
        "selection": (
            "Largest common per-mode validation false-alarm allowance whose "
            "five-mode union remains within the validation union budget."
        ),
        "alarm_rule": "score >= threshold; union = any of five modes",
        "cost_assumption": {
            "devices": DEVICES,
            "test_prevalence": TEST_PREVALENCE,
            "miss_to_false_cost_ratio": MISS_TO_FALSE_COST_RATIO,
        },
        "source_sha256": {
            "validation_predictions": sha256(OUT / "validation_predictions.csv"),
            "test_predictions": sha256(OUT / "test_predictions.csv"),
            "freeze_manifest": sha256(OUT / "freeze_manifest.json"),
        },
    }
    (OUT / "system_alarm_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("PASS: system-level union-FPR thresholds calibrated and evaluated")


if __name__ == "__main__":
    main()
