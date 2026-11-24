"""Prespecified robustness checks for Workstream D.

Run after D tune/evaluate-test/noise:
    python -m Group19_D.extended_analysis

The script does not refit or retune. It adds:
1. a row-bootstrap interval for per-mode and aggregate test metrics;
2. macro-AP sensitivity to the extremely sparse RNF mode;
3. an explicit rank/quantile/fixed-probability threshold comparison.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from Group19_A.pipeline import MODES, load_split
from Group19_D.run import OUT, read_config, verify_freeze

BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20261124
FIXED_PROBABILITY_THRESHOLD = 0.05
INTERVAL_QUANTILES = (0.025, 0.975)
INTERPRETABLE_MODES = ["TWF", "HDF", "PWF", "OSF"]


def binary_metric_values(y_mode, score, alarm, total_failure) -> dict:
    y_mode = np.asarray(y_mode, dtype=int)
    alarm = np.asarray(alarm, dtype=bool)
    total_failure = np.asarray(total_failure, dtype=int)
    tp = int(np.sum((y_mode == 1) & alarm))
    fp = int(np.sum((y_mode == 0) & alarm))
    fn = int(np.sum((y_mode == 1) & ~alarm))
    positives = int(y_mode.sum())
    negatives = int((y_mode == 0).sum())
    normal = total_failure == 0
    normal_fp = int(np.sum(normal & alarm))
    ap = (
        float(average_precision_score(y_mode, score))
        if positives > 0 and negatives > 0
        else np.nan
    )
    recall = tp / positives if positives else np.nan
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator else np.nan
    return {
        "average_precision": ap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": positives,
        "normal_fp": normal_fp,
        "normal_denominator": int(normal.sum()),
        "normal_fpr": normal_fp / int(normal.sum()) if normal.any() else np.nan,
    }


def interval(values) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    if not len(valid):
        return np.nan, np.nan, 0
    low, high = np.quantile(valid, INTERVAL_QUANTILES)
    return float(low), float(high), int(len(valid))


def load_partition(partition: str):
    config = read_config()
    predictions = pd.read_csv(
        OUT / f"{partition}_predictions.csv",
        float_precision="round_trip",
    )
    _, total_failure, labels = load_split(
        partition,
        allow_test=partition == "test",
    )
    aligned_index = total_failure.index.sort_values()
    total_failure = total_failure.loc[aligned_index]
    labels = labels.loc[aligned_index]
    prediction_index = (
        predictions.drop_duplicates("row_index")
        .set_index("row_index")
        .sort_index()
    )
    if not prediction_index.index.equals(total_failure.index):
        raise ValueError(f"{partition}: prediction row order differs from A")
    result = {}
    for feature_set in config["feature_sets"]:
        frame = predictions[predictions.feature_set == feature_set]
        score_matrix = np.column_stack([
            frame[frame["mode"] == mode].set_index("row_index").loc[total_failure.index, "score"]
            for mode in MODES
        ])
        threshold_matrix = np.column_stack([
            frame[frame["mode"] == mode].set_index("row_index").loc[total_failure.index, "threshold"]
            for mode in MODES
        ])
        label_matrix = labels[MODES].loc[total_failure.index].to_numpy(dtype=int)
        alarm_matrix = score_matrix >= threshold_matrix
        result[feature_set] = {
            "row_index": total_failure.index.to_numpy(),
            "total_failure": total_failure.to_numpy(dtype=int),
            "labels": label_matrix,
            "scores": score_matrix,
            "alarms": alarm_matrix,
            "thresholds": threshold_matrix[0],
        }
    return result


def aggregate_values(labels, scores, alarms, total_failure) -> dict:
    supports = labels.sum(axis=0)
    mode_aps = np.asarray([
        average_precision_score(labels[:, i], scores[:, i])
        if 0 < labels[:, i].sum() < len(labels)
        else np.nan
        for i in range(len(MODES))
    ], dtype=float)
    interpretable_mask = np.asarray([mode in INTERPRETABLE_MODES for mode in MODES])
    weights = supports.astype(float)
    support_weighted_ap_all = (
        float(np.average(mode_aps, weights=weights))
        if weights.sum() > 0 and np.isfinite(mode_aps).all()
        else np.nan
    )
    support_weighted_ap_without_rnf = (
        float(np.average(
            mode_aps[interpretable_mask],
            weights=weights[interpretable_mask],
        ))
        if weights[interpretable_mask].sum() > 0
        and np.isfinite(mode_aps[interpretable_mask]).all()
        else np.nan
    )
    normal = total_failure == 0
    union_alarm = alarms[normal].any(axis=1)
    return {
        "macro_ap_all_five": float(np.mean(mode_aps)),
        "macro_ap_excluding_rnf": float(np.mean(mode_aps[interpretable_mask])),
        "support_weighted_ap_all_five": support_weighted_ap_all,
        "support_weighted_ap_excluding_rnf": support_weighted_ap_without_rnf,
        "any_mode_normal_fp": int(union_alarm.sum()),
        "any_mode_normal_denominator": int(normal.sum()),
        "any_mode_normal_fpr": float(union_alarm.mean()),
    }


def bootstrap(partitions) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for feature_set, data in partitions.items():
        total = data["total_failure"]
        positive_rows = np.flatnonzero(total == 1)
        negative_rows = np.flatnonzero(total == 0)
        if not len(positive_rows) or not len(negative_rows):
            raise ValueError("Bootstrap requires both classes")
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for repeat in range(BOOTSTRAP_REPEATS):
            sampled = np.concatenate([
                rng.choice(positive_rows, len(positive_rows), replace=True),
                rng.choice(negative_rows, len(negative_rows), replace=True),
            ])
            labels = data["labels"][sampled]
            scores = data["scores"][sampled]
            alarms = data["alarms"][sampled]
            total_sampled = data["total_failure"][sampled]
            aggregate = aggregate_values(labels, scores, alarms, total_sampled)
            row = {
                "repeat": repeat,
                "feature_set": feature_set,
                **aggregate,
            }
            for mode_index, mode in enumerate(MODES):
                metrics = binary_metric_values(
                    labels[:, mode_index],
                    scores[:, mode_index],
                    alarms[:, mode_index],
                    total_sampled,
                )
                row[f"{mode}_ap"] = metrics["average_precision"]
                row[f"{mode}_recall"] = metrics["recall"]
                row[f"{mode}_f1"] = metrics["f1"]
                row[f"{mode}_normal_fpr"] = metrics["normal_fpr"]
            rows.append(row)
            if (repeat + 1) % 250 == 0:
                print(f"bootstrap {repeat + 1}/{BOOTSTRAP_REPEATS}", flush=True)
    replicates = pd.DataFrame(rows)
    replicates.to_csv(
        OUT / "bootstrap_replicates.csv",
        index=False,
        float_format="%.17g",
    )

    point_frames = []
    for feature_set, data in partitions.items():
        aggregate = aggregate_values(
            data["labels"],
            data["scores"],
            data["alarms"],
            data["total_failure"],
        )
        for metric, value in aggregate.items():
            if metric in ("any_mode_normal_fp", "any_mode_normal_denominator"):
                continue
            point_frames.append({
                "feature_set": feature_set,
                "scope": "aggregate",
                "mode": "ALL",
                "metric": metric,
                "point": value,
            })
        for mode_index, mode in enumerate(MODES):
            metrics = binary_metric_values(
                data["labels"][:, mode_index],
                data["scores"][:, mode_index],
                data["alarms"][:, mode_index],
                data["total_failure"],
            )
            for metric in ("average_precision", "recall", "f1", "normal_fpr"):
                point_frames.append({
                    "feature_set": feature_set,
                    "scope": "mode",
                    "mode": mode,
                    "metric": metric,
                    "point": metrics[metric],
                })

    interval_rows = []
    for point_row in point_frames:
        feature_set = point_row["feature_set"]
        scope = point_row["scope"]
        mode = point_row["mode"]
        metric = point_row["metric"]
        replicate = replicates[replicates.feature_set == feature_set]
        if scope == "aggregate":
            values = replicate[metric]
        else:
            values = replicate[f"{mode}_{metric_map(metric)}"]
        low, high, valid = interval(values)
        support = (
            int(partitions[feature_set]["labels"][:, MODES.index(mode)].sum())
            if scope == "mode"
            else int(len(partitions[feature_set]["total_failure"]))
        )
        interval_rows.append({
            **point_row,
            "ci_low": low,
            "ci_high": high,
            "valid_repeats": valid,
            "total_repeats": BOOTSTRAP_REPEATS,
            "support": support,
            "interpretation": (
                "insufficient_support" if mode == "RNF" else "reportable"
            ),
        })
    intervals = pd.DataFrame(interval_rows)
    intervals.to_csv(
        OUT / "bootstrap_intervals.csv",
        index=False,
        float_format="%.17g",
    )
    return replicates, intervals


def metric_map(metric: str) -> str:
    return {
        "average_precision": "ap",
        "recall": "recall",
        "f1": "f1",
        "normal_fpr": "normal_fpr",
    }[metric]


def macro_sensitivity(intervals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feature_set, metric), group in intervals[
        intervals.scope == "aggregate"
    ].groupby(["feature_set", "metric"], sort=False):
        if metric not in (
            "macro_ap_all_five",
            "macro_ap_excluding_rnf",
            "support_weighted_ap_all_five",
            "support_weighted_ap_excluding_rnf",
        ):
            continue
        row = group.iloc[0]
        rows.append({
            "feature_set": feature_set,
            "metric": metric,
            "point": row.point,
            "ci_low": row.ci_low,
            "ci_high": row.ci_high,
            "valid_repeats": row.valid_repeats,
            "interpretation": (
                "includes_sparse_rnf"
                if metric in ("macro_ap_all_five", "support_weighted_ap_all_five")
                else "excluding_rnf_sensitivity"
            ),
        })
    result = pd.DataFrame(rows)
    result.to_csv(
        OUT / "macro_ap_sensitivity.csv",
        index=False,
        float_format="%.17g",
    )
    return result


def threshold_values(labels, scores, threshold, total_failure) -> dict:
    return binary_metric_values(
        labels,
        scores,
        np.asarray(scores) >= float(threshold),
        total_failure,
    )


def threshold_calibration(validation, test) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    anchors = []
    validation_base = next(iter(validation.values()))
    normal = validation_base["total_failure"] == 0
    normal_count = int(normal.sum())
    allowed = int(np.floor(0.05 * normal_count))
    quantile_probability = 1 - allowed / normal_count
    for feature_set in sorted(validation):
        validation_data = validation[feature_set]
        test_data = test[feature_set]
        for mode_index, mode in enumerate(MODES):
            normal_scores = np.sort(
                validation_data["scores"][normal, mode_index]
            )
            raw_quantile = float(np.quantile(
                normal_scores,
                quantile_probability,
                method="higher",
            ))
            tie_adjusted_quantile = float(np.nextafter(raw_quantile, np.inf))
            current_threshold = float(
                validation_data["thresholds"][mode_index]
            )
            methods = {
                "rank_budget": current_threshold,
                "empirical_quantile_tie_adjusted": tie_adjusted_quantile,
                "fixed_probability_0_05": FIXED_PROBABILITY_THRESHOLD,
            }
            anchors.append({
                "feature_set": feature_set,
                "mode": mode,
                "validation_normal_count": normal_count,
                "allowed_validation_fp": allowed,
                "rank_cutoff_excluding_fp": allowed + 1,
                "quantile_probability": quantile_probability,
                "raw_empirical_quantile": raw_quantile,
                "current_threshold": current_threshold,
                "quantile_threshold_tie_adjusted": tie_adjusted_quantile,
                "current_equals_quantile_adjusted": bool(
                    current_threshold == tie_adjusted_quantile
                ),
                "numeric_note": (
                    "Near-zero values follow from rank calibration on discrete "
                    "random-forest probabilities; the operating point is the rank, "
                    "not the absolute probability scale."
                ),
                "test_support": int(test_data["labels"][:, mode_index].sum()),
            })
            for method, threshold in methods.items():
                for partition, data in (
                    ("validation", validation_data),
                    ("test", test_data),
                ):
                    metrics = threshold_values(
                        data["labels"][:, mode_index],
                        data["scores"][:, mode_index],
                        threshold,
                        data["total_failure"],
                    )
                    rows.append({
                        "feature_set": feature_set,
                        "mode": mode,
                        "method": method,
                        "partition": partition,
                        "threshold": threshold,
                        **metrics,
                    })
    detail = pd.DataFrame(rows)
    detail.to_csv(
        OUT / "threshold_calibration.csv",
        index=False,
        float_format="%.17g",
    )
    anchors_frame = pd.DataFrame(anchors)
    anchors_frame.to_csv(
        OUT / "threshold_anchors.csv",
        index=False,
        float_format="%.17g",
    )
    return detail, anchors_frame


def threshold_system_summary(validation, test, anchors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    methods = {
        "rank_budget": "current_threshold",
        "empirical_quantile_tie_adjusted": "quantile_threshold_tie_adjusted",
        "fixed_probability_0_05": None,
    }
    for feature_set in sorted(validation):
        anchor = anchors[anchors.feature_set == feature_set].set_index("mode")
        for method, anchor_column in methods.items():
            if anchor_column is None:
                thresholds = np.full(len(MODES), FIXED_PROBABILITY_THRESHOLD)
            else:
                thresholds = anchor.loc[MODES, anchor_column].to_numpy(dtype=float)
            for partition, data in (
                ("validation", validation[feature_set]),
                ("test", test[feature_set]),
            ):
                alarms = data["scores"] >= thresholds
                total = data["total_failure"]
                y = data["labels"]
                mode_f1 = []
                mode_recall = []
                mode_precision = []
                tp_all = fp_all = fn_all = 0
                for mode_index in range(len(MODES)):
                    metrics = binary_metric_values(
                        y[:, mode_index],
                        data["scores"][:, mode_index],
                        alarms[:, mode_index],
                        total,
                    )
                    mode_f1.append(metrics["f1"])
                    mode_recall.append(metrics["recall"])
                    mode_precision.append(metrics["precision"])
                    tp_all += metrics["tp"]
                    fp_all += metrics["fp"]
                    fn_all += metrics["fn"]
                normal = total == 0
                failure = total == 1
                any_normal = alarms[normal].any(axis=1)
                any_failure = alarms[failure].any(axis=1)
                rows.append({
                    "feature_set": feature_set,
                    "method": method,
                    "partition": partition,
                    "macro_precision": float(np.nanmean(mode_precision)),
                    "macro_recall": float(np.nanmean(mode_recall)),
                    "macro_f1": float(np.nanmean(mode_f1)),
                    "micro_f1": (
                        2 * tp_all / (2 * tp_all + fp_all + fn_all)
                        if 2 * tp_all + fp_all + fn_all
                        else np.nan
                    ),
                    "any_mode_normal_fp": int(any_normal.sum()),
                    "any_mode_normal_denominator": int(normal.sum()),
                    "any_mode_normal_fpr": float(any_normal.mean()),
                    "any_mode_failure_recall": float(any_failure.mean()),
                })
    result = pd.DataFrame(rows)
    result.to_csv(
        OUT / "threshold_system_summary.csv",
        index=False,
        float_format="%.17g",
    )
    return result


def figures(intervals: pd.DataFrame, threshold_detail: pd.DataFrame) -> None:
    colors = {"raw": "#6F6F6F", "engineered": "#5B0F78"}
    mode_positions = np.arange(len(MODES))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for offset, feature_set in ((-width / 2, "raw"), (width / 2, "engineered")):
        frame = intervals[
            (intervals.feature_set == feature_set)
            & (intervals.scope == "mode")
            & (intervals.metric == "average_precision")
        ].set_index("mode").loc[MODES]
        low = frame.point - frame.ci_low
        high = frame.ci_high - frame.point
        ax.errorbar(
            mode_positions + offset,
            frame.point,
            yerr=np.vstack([low, high]),
            marker="o",
            capsize=4,
            linewidth=2,
            color=colors[feature_set],
            label=feature_set,
        )
    ax.set_xticks(mode_positions, [f"{m} (n={int(intervals[(intervals.scope == 'mode') & (intervals.feature_set == 'raw') & (intervals['mode'] == m) & (intervals.metric == 'average_precision')].support.iloc[0])})" for m in MODES])
    ax.set_ylabel("Test average precision")
    ax.set_ylim(-0.02, 1.04)
    ax.set_title("Bootstrap 95% intervals: per-mode AP (RNF remains n=2)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "bootstrap_intervals.png", dpi=180)
    plt.close(fig)

    test = threshold_detail[threshold_detail.partition == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
    methods = [
        "rank_budget",
        "empirical_quantile_tie_adjusted",
        "fixed_probability_0_05",
    ]
    colors_method = ["#5B0F78", "#B47CC7", "#B07A00"]
    for axis, metric, title in (
        (axes[0], "recall", "Test recall"),
        (axes[1], "normal_fpr", "Test total-normal FPR"),
    ):
        x = np.arange(len(MODES))
        for offset, method, color in zip(
            (-0.26, 0.0, 0.26),
            methods,
            colors_method,
        ):
            values = [
                test[
                    (test.feature_set == "engineered")
                    & (test["mode"] == mode)
                    & (test.method == method)
                ][metric].iloc[0]
                for mode in MODES
            ]
            axis.bar(x + offset, values, width=0.24, color=color, label=method)
        axis.set_xticks(x, MODES)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 1.05)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Engineered inputs: rank budget vs tie-adjusted quantile vs fixed 0.05")
    fig.savefig(OUT / "threshold_calibration.png", dpi=180)
    plt.close(fig)


def main() -> None:
    verify_freeze()
    partitions = {
        "validation": load_partition("validation"),
        "test": load_partition("test"),
    }
    _, intervals = bootstrap(partitions["test"])
    macro_sensitivity(intervals)
    threshold_detail, anchors = threshold_calibration(
        partitions["validation"],
        partitions["test"],
    )
    threshold_system_summary(
        partitions["validation"],
        partitions["test"],
        anchors,
    )
    figures(intervals, threshold_detail)
    def sha256(name: str) -> str:
        return hashlib.sha256((OUT / name).read_bytes()).hexdigest()

    manifest = {
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_strata": ["Machine failure == 0", "Machine failure == 1"],
        "interval": "2.5th and 97.5th percentiles",
        "rnf_interpretation": "n=2; report intervals but do not rank",
        "macro_ap_metrics": [
            "macro_ap_all_five",
            "macro_ap_excluding_rnf",
            "support_weighted_ap",
        ],
        "threshold_comparison": [
            "rank_budget",
            "empirical_quantile_tie_adjusted",
            "fixed_probability_0_05",
        ],
        "current_threshold_equals_quantile_adjusted": bool(
            anchors["current_equals_quantile_adjusted"].all()
        ),
        "source_hashes": {
            "validation_predictions": sha256("validation_predictions.csv"),
            "test_predictions": sha256("test_predictions.csv"),
            "freeze_manifest": sha256("freeze_manifest.json"),
        },
    }
    (OUT / "extended_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "PASS: 2,000-replicate bootstrap, RNF/macro-AP sensitivity, and "
        "threshold calibration comparison completed"
    )


if __name__ == "__main__":
    main()
