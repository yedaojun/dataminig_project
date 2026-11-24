"""Translate frozen operating points into per-1,000-device alarm costs.

Run from the repository root:
    python -m Group19_D.alarm_cost_analysis

One test record is treated as one device inspection. Counts are expected
values, not real fleet observations. Cost is expressed in units where one
false-positive review costs one unit and C_FN/C_FP is varied.
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

from Group19_D.run import OUT

REPO = Path(__file__).resolve().parent.parent
C_RESULTS = REPO / "Group19_C" / "results"
B_RESULTS = REPO / "Group19_B" / "results"
DEVICES = 1000
TEST_PREVALENCE = 51 / 1500
PREVALENCES = (0.01, TEST_PREVALENCE, 0.05, 0.10)
FALSE_TO_MISS_COST_RATIOS = (1, 5, 10, 20, 50)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def c_candidates():
    validation = pd.read_csv(
        C_RESULTS / "validation_metrics.csv",
        float_precision="round_trip",
    )
    engineered = validation[validation.feature_set == "engineered"].sort_values(
        "average_precision",
        ascending=False,
        kind="stable",
    ).iloc[0]
    raw = validation[validation.feature_set == "raw"].sort_values(
        "average_precision",
        ascending=False,
        kind="stable",
    ).iloc[0]
    return engineered, raw


def b_candidate():
    metric_paths = list(
        (B_RESULTS / "final_test").glob("*/metrics.csv")
    )
    if len(metric_paths) != 1:
        raise ValueError("Expected one B final-test metrics file")
    metrics = pd.read_csv(metric_paths[0], float_precision="round_trip")
    return metrics[metrics.feature_set == "raw"].sort_values(
        "average_precision",
        ascending=False,
        kind="stable",
    ).iloc[0]


def operating_points() -> pd.DataFrame:
    c_engineered, c_raw = c_candidates()
    c_test = pd.read_csv(
        C_RESULTS / "test_metrics.csv",
        float_precision="round_trip",
    ).set_index("candidate")
    c_alarms = pd.read_csv(
        C_RESULTS / "alarm_budgets.csv",
        float_precision="round_trip",
    )
    b = b_candidate()
    d = pd.read_csv(
        OUT / "threshold_system_summary.csv",
        float_precision="round_trip",
    )
    rows = []

    engineered_f1 = c_test.loc[c_engineered.candidate]
    rows.append({
        "operating_point": "C_engineered_RF_F1",
        "model_family": "C supervised binary",
        "feature_set": "engineered",
        "threshold_source": "validation positive-class F1",
        "test_recall": float(engineered_f1.recall),
        "test_normal_fpr": float(
            engineered_f1.fp / (engineered_f1.fp + engineered_f1.tn)
        ),
        "test_average_precision": float(engineered_f1.average_precision),
        "test_f1": float(engineered_f1.f1),
        "test_support": int(engineered_f1.tp + engineered_f1.fn),
    })
    for budget in (0.01, 0.02, 0.05):
        row = c_alarms[
            (c_alarms.model == "C_engineered_validation_best")
            & (c_alarms.partition == "test")
            & np.isclose(c_alarms.target_fpr, budget)
        ].iloc[0]
        rows.append({
            "operating_point": f"C_engineered_RF_{budget:.0%}_validation_fpr",
            "model_family": "C supervised binary",
            "feature_set": "engineered",
            "threshold_source": f"validation-normal FPR <= {budget:.0%}",
            "test_recall": float(row.recall),
            "test_normal_fpr": float(row.actual_fpr),
            "test_average_precision": np.nan,
            "test_f1": np.nan,
            "test_support": int(engineered_f1.tp + engineered_f1.fn),
        })

    raw_f1 = c_test.loc[c_raw.candidate]
    rows.append({
        "operating_point": "C_raw_XGB_F1",
        "model_family": "C supervised binary",
        "feature_set": "raw",
        "threshold_source": "validation positive-class F1",
        "test_recall": float(raw_f1.recall),
        "test_normal_fpr": float(raw_f1.fp / (raw_f1.fp + raw_f1.tn)),
        "test_average_precision": float(raw_f1.average_precision),
        "test_f1": float(raw_f1.f1),
        "test_support": int(raw_f1.tp + raw_f1.fn),
    })
    rows.append({
        "operating_point": "B_raw_LOF_F1",
        "model_family": "B feature-only scoring",
        "feature_set": "raw",
        "threshold_source": "validation positive-class F1",
        "test_recall": float(b.recall),
        "test_normal_fpr": float(b.fp / (b.fp + b.tn)),
        "test_average_precision": float(b.average_precision),
        "test_f1": float(b.f1),
        "test_support": int(b.tp + b.fn),
    })

    for feature_set in ("raw", "engineered"):
        row = d[
            (d.feature_set == feature_set)
            & (d.partition == "test")
            & (d.method == "rank_budget")
        ].iloc[0]
        rows.append({
            "operating_point": f"D_{feature_set}_per_mode_5pct_union",
            "model_family": "D five-mode diagnosis",
            "feature_set": feature_set,
            "threshold_source": "5% FPR per mode; union alarms",
            "test_recall": float(row.any_mode_failure_recall),
            "test_normal_fpr": float(row.any_mode_normal_fpr),
            "test_average_precision": np.nan,
            "test_f1": float(row.macro_f1),
            "test_support": 51,
        })
    return pd.DataFrame(rows)


def expand_cost_table(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for point in points.itertuples():
        for prevalence in PREVALENCES:
            failures = DEVICES * prevalence
            normals = DEVICES - failures
            for ratio in FALSE_TO_MISS_COST_RATIOS:
                tp = failures * point.test_recall
                fn = failures - tp
                fp = normals * point.test_normal_fpr
                alarms = tp + fp
                cost = fp + ratio * fn
                rows.append({
                    "operating_point": point.operating_point,
                    "model_family": point.model_family,
                    "feature_set": point.feature_set,
                    "threshold_source": point.threshold_source,
                    "devices": DEVICES,
                    "prevalence": prevalence,
                    "expected_failures": failures,
                    "expected_tp": tp,
                    "expected_fp": fp,
                    "expected_fn": fn,
                    "expected_alarms": alarms,
                    "alarms_per_detected_failure": (
                        alarms / tp if tp > 0 else np.nan
                    ),
                    "precision": tp / alarms if alarms > 0 else np.nan,
                    "miss_to_false_cost_ratio": ratio,
                    "expected_cost_units": cost,
                    "cost_units_per_detected_failure": (
                        cost / tp if tp > 0 else np.nan
                    ),
                })
    return pd.DataFrame(rows)


def figures(points: pd.DataFrame, costs: pd.DataFrame) -> None:
    actual = costs[np.isclose(costs.prevalence, TEST_PREVALENCE)]
    key_points = [
        "B_raw_LOF_F1",
        "C_raw_XGB_F1",
        "C_engineered_RF_F1",
        "C_engineered_RF_1%_validation_fpr",
        "D_raw_per_mode_5pct_union",
        "D_engineered_per_mode_5pct_union",
    ]
    actual_one = actual[
        (actual.miss_to_false_cost_ratio == 10)
        & actual.operating_point.isin(key_points)
    ].set_index("operating_point").loc[key_points]
    labels = [
        "B LOF\nF1",
        "C raw XGB\nF1",
        "C eng RF\nF1",
        "C eng RF\n1% FPR",
        "D raw\nunion",
        "D eng\nunion",
    ]
    x = np.arange(len(key_points))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    axes[0].bar(x - 0.2, actual_one.expected_fp, 0.4, color="#B07A00", label="False alarms")
    axes[0].bar(x + 0.2, actual_one.expected_tp, 0.4, color="#5B0F78", label="Detected failures")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Expected count per 1,000 devices")
    axes[0].set_title("Alarms at the test failure prevalence (3.4%)")
    axes[1].bar(x, actual_one.expected_cost_units, color=["#777777", "#777777", "#5B0F78", "#B47CC7", "#6F6F6F", "#5B0F78"])
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Cost units per 1,000 devices")
    axes[1].set_title("Relative cost when one miss = 10 false alarms")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Operational interpretation of frozen alarm thresholds")
    fig.savefig(OUT / "alarm_cost_per_1000.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    selected = [
        "C_engineered_RF_F1",
        "C_engineered_RF_1%_validation_fpr",
        "B_raw_LOF_F1",
        "D_engineered_per_mode_5pct_union",
    ]
    style = {
        "C_engineered_RF_F1": ("#5B0F78", "D", "C eng RF F1"),
        "C_engineered_RF_1%_validation_fpr": ("#B47CC7", "s", "C eng RF 1%"),
        "B_raw_LOF_F1": ("#777777", "o", "B raw LOF F1"),
        "D_engineered_per_mode_5pct_union": ("#B07A00", "^", "D eng union"),
    }
    actual = actual[np.isclose(actual.prevalence, TEST_PREVALENCE)]
    for point in selected:
        frame = actual[actual.operating_point == point].sort_values(
            "miss_to_false_cost_ratio"
        )
        color, marker, label = style[point]
        ax.plot(
            frame.miss_to_false_cost_ratio,
            frame.expected_cost_units,
            marker=marker,
            linewidth=2,
            color=color,
            label=label,
        )
    ax.set_xlabel("Cost of one missed failure / cost of one false alarm")
    ax.set_ylabel("Cost units per 1,000 devices")
    ax.set_title("Cost sensitivity at 3.4% failure prevalence")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "alarm_cost_sensitivity.png", dpi=180)
    plt.close(fig)


def write_manifest(points: pd.DataFrame) -> None:
    manifest = {
        "unit": "one test record = one device inspection cycle",
        "devices": DEVICES,
        "prevalences": list(PREVALENCES),
        "test_prevalence": TEST_PREVALENCE,
        "miss_to_false_cost_ratios": list(FALSE_TO_MISS_COST_RATIOS),
        "cost_units": (
            "expected_cost = FP * 1 + FN * (C_FN / C_FP)"
        ),
        "operating_points": points.operating_point.tolist(),
        "source_sha256": {
            "C_test_metrics": sha256(C_RESULTS / "test_metrics.csv"),
            "C_alarm_budgets": sha256(C_RESULTS / "alarm_budgets.csv"),
            "D_threshold_system_summary": sha256(
                OUT / "threshold_system_summary.csv"
            ),
        },
    }
    (OUT / "alarm_cost_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    points = operating_points()
    costs = expand_cost_table(points)
    points.to_csv(
        OUT / "alarm_operating_points.csv",
        index=False,
        float_format="%.17g",
    )
    costs.to_csv(
        OUT / "alarm_cost_scenarios.csv",
        index=False,
        float_format="%.17g",
    )
    actual = costs[np.isclose(costs.prevalence, TEST_PREVALENCE)].copy()
    actual.to_csv(
        OUT / "alarm_cost_actual_prevalence.csv",
        index=False,
        float_format="%.17g",
    )
    figures(points, costs)
    write_manifest(points)
    print(
        "PASS: per-1,000-device alarm and cost scenarios completed"
    )


if __name__ == "__main__":
    main()
