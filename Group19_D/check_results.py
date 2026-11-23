"""Independent invariant checks for Group 19 D deliverables.

Run from the repository root after tune, evaluate-test, and noise:
    python -m Group19_D.check_results
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from Group19_A.pipeline import MODES, load_split
from Group19_D.run import CONFIG, FREEZE_FILE, MODEL_FILE, OUT, read_config, verify_freeze


def recompute_metrics(y, scores, threshold, total_failure) -> dict:
    y = np.asarray(y)
    total_failure = np.asarray(total_failure)
    alarm = np.asarray(scores) >= float(threshold)
    tp = int(np.sum((y == 1) & alarm))
    fp = int(np.sum((y == 0) & alarm))
    tn = int(np.sum((y == 0) & ~alarm))
    fn = int(np.sum((y == 1) & ~alarm))
    normal = total_failure == 0
    mode_negative = y == 0
    normal_fp = int(np.sum(normal & alarm))
    mode_negative_fp = int(np.sum(mode_negative & alarm))
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "threshold": float(threshold),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "support": int(y.sum()),
        "normal_fp": normal_fp,
        "normal_denominator": int(normal.sum()),
        "normal_fpr": normal_fp / int(normal.sum()),
        "mode_negative_fp": mode_negative_fp,
        "mode_negative_denominator": int(mode_negative.sum()),
        "mode_negative_fpr": mode_negative_fp / int(mode_negative.sum()),
    }


def check_partition(partition: str, config: dict) -> pd.DataFrame:
    predictions = pd.read_csv(
        OUT / f"{partition}_predictions.csv",
        float_precision="round_trip",
    )
    metrics = pd.read_csv(
        OUT / f"{partition}_metrics.csv",
        float_precision="round_trip",
    )
    expected_rows = 1500
    expected_total = len(config["feature_sets"]) * len(MODES) * expected_rows
    assert len(predictions) == expected_total
    assert len(metrics) == len(config["feature_sets"]) * len(MODES)
    assert set(predictions["mode"]) == set(MODES)
    assert set(metrics["mode"]) == set(MODES)
    expected_x, expected_y, expected_Y = load_split(
        partition,
        allow_test=partition == "test",
    )
    assert expected_x.index.is_unique and len(expected_x) == expected_rows
    for feature_set in config["feature_sets"]:
        for mode in MODES:
            frame = predictions[
                (predictions.feature_set == feature_set)
                & (predictions["mode"] == mode)
            ].set_index("row_index")
            assert len(frame) == expected_rows and frame.index.is_unique
            assert set(frame.index) == set(expected_x.index)
            assert set(frame.index) == set(expected_y.index)
            assert np.array_equal(
                frame.y_true.to_numpy(),
                expected_Y[mode].loc[frame.index].to_numpy(),
            )
            assert np.array_equal(
                frame.total_failure.to_numpy(),
                expected_y.loc[frame.index].to_numpy(),
            )
            alarm = (frame.score.to_numpy() >= frame.threshold.iloc[0]).astype(int)
            assert np.array_equal(alarm, frame.alarm.to_numpy())
            row = metrics[
                (metrics.feature_set == feature_set)
                & (metrics["mode"] == mode)
            ]
            assert len(row) == 1
            row = row.iloc[0]
            recomputed = recompute_metrics(
                frame.y_true,
                frame.score,
                row.threshold,
                frame.total_failure,
            )
            for field, value in recomputed.items():
                if isinstance(value, float):
                    assert np.isclose(value, float(row[field]), rtol=0, atol=1e-12), (
                        partition,
                        feature_set,
                        mode,
                        field,
                    )
                else:
                    assert int(value) == int(row[field])
    return predictions


def check_candidate_selection(config: dict) -> None:
    candidates = pd.read_csv(OUT / "candidates.csv")
    assert len(candidates) == (
        len(config["feature_sets"]) * len(config["candidates"])
    )
    for feature_set in config["feature_sets"]:
        group = candidates[candidates.feature_set == feature_set]
        group = group.sort_values("candidate_order")
        selected = group[group.selected == 1]
        assert len(selected) == 1
        winner = selected.iloc[0]
        assert np.isclose(
            winner.validation_macro_average_precision,
            group.validation_macro_average_precision.max(),
            rtol=0,
            atol=0,
        )
        # `max` over grid order preserves the first candidate on exact ties.
        first_max = group.iloc[0]
        for _, candidate in group.iloc[1:].iterrows():
            if (
                candidate.validation_macro_average_precision
                > first_max.validation_macro_average_precision
            ):
                first_max = candidate
        assert winner.candidate == first_max.candidate


def check_thresholds(manifest: dict, config: dict) -> None:
    validation = pd.read_csv(
        OUT / "validation_predictions.csv",
        float_precision="round_trip",
    )
    validation_metrics = pd.read_csv(
        OUT / "validation_metrics.csv",
        float_precision="round_trip",
    )
    normal_count = int((validation.drop_duplicates("row_index").total_failure == 0).sum())
    allowed = int(np.floor(
        config["diagnostic_threshold"]["target_false_positive_rate"] * normal_count
    ))
    assert normal_count == 1449 and allowed == 72
    for feature_set in config["feature_sets"]:
        for mode in MODES:
            frame = validation[
                (validation.feature_set == feature_set)
                & (validation["mode"] == mode)
            ]
            threshold = manifest["selected"][feature_set]["thresholds"][mode]
            assert np.isclose(frame.threshold.iloc[0], threshold, rtol=0, atol=0)
            fp = int(np.sum(
                (frame.total_failure == 0)
                & (frame.score >= threshold)
            ))
            assert fp <= allowed
            metric_row = validation_metrics[
                (validation_metrics.feature_set == feature_set)
                & (validation_metrics["mode"] == mode)
            ]
            assert len(metric_row) == 1
            assert int(metric_row.normal_fp.iloc[0]) == fp


def check_aggregates() -> None:
    for partition in ("validation", "test"):
        predictions = pd.read_csv(
            OUT / f"{partition}_predictions.csv",
            float_precision="round_trip",
        )
        aggregates = pd.read_csv(
            OUT / f"{partition}_aggregates.csv",
            float_precision="round_trip",
        )
        for feature_set in ("raw", "engineered"):
            frame = predictions[predictions.feature_set == feature_set]
            row = aggregates[aggregates.feature_set == feature_set].iloc[0]
            mode_metrics = []
            for mode in MODES:
                group = frame[frame["mode"] == mode]
                mode_metrics.append(recompute_metrics(
                    group.y_true,
                    group.score,
                    group.threshold.iloc[0],
                    group.total_failure,
                ))
            assert np.isclose(
                row.macro_average_precision,
                np.mean([metric["average_precision"] for metric in mode_metrics]),
                rtol=0,
                atol=1e-12,
            )
            assert np.isclose(
                row.macro_f1,
                np.mean([metric["f1"] for metric in mode_metrics]),
                rtol=0,
                atol=1e-12,
            )
            tp = sum(metric["tp"] for metric in mode_metrics)
            fp = sum(metric["fp"] for metric in mode_metrics)
            fn = sum(metric["fn"] for metric in mode_metrics)
            assert np.isclose(
                row.micro_f1,
                2 * tp / (2 * tp + fp + fn),
                rtol=0,
                atol=1e-12,
            )
            assert (int(row.micro_tp), int(row.micro_fp), int(row.micro_fn)) == (
                tp,
                fp,
                fn,
            )


def check_cases_and_support() -> None:
    cases = pd.read_csv(OUT / "cases.csv")
    support = pd.read_csv(OUT / "mode_support.csv")
    predictions = pd.read_csv(
        OUT / "test_predictions.csv",
        float_precision="round_trip",
    )
    assert set(cases.case_type) == {"detected", "false_positive", "missed"}
    assert not cases.duplicated(["feature_set", "mode", "case_type"]).any()
    for feature_set in ("raw", "engineered"):
        for mode in MODES:
            frame = predictions[
                (predictions.feature_set == feature_set)
                & (predictions["mode"] == mode)
            ]
            masks = {
                "detected": (frame.y_true == 1) & (frame.alarm == 1),
                "false_positive": (frame.y_true == 0) & (frame.alarm == 1),
                "missed": (frame.y_true == 1) & (frame.alarm == 0),
            }
            for case_type, mask in masks.items():
                candidates = frame[mask]
                selected = cases[
                    (cases.feature_set == feature_set)
                    & (cases["mode"] == mode)
                    & (cases.case_type == case_type)
                ]
                assert len(selected) == (0 if candidates.empty else 1)
                if not candidates.empty:
                    assert np.isclose(
                        selected.score.iloc[0],
                        candidates.score.max(),
                        rtol=0,
                        atol=1e-15,
                    )
                    assert int(selected.row_index.iloc[0]) == int(
                        candidates.sort_values("score", ascending=False).iloc[0].row_index
                    )
    test_mode_support = support[support.partition == "test"].set_index("mode")["support"]
    assert test_mode_support[list(MODES)].to_dict() == {
        "TWF": 8,
        "HDF": 18,
        "PWF": 13,
        "OSF": 15,
        "RNF": 2,
    }
    assert int(test_mode_support["TOTAL_FAILURE"]) == 51


def check_noise(config: dict) -> None:
    detail = pd.read_csv(OUT / "noise_repeats.csv", float_precision="round_trip")
    summary = pd.read_csv(OUT / "noise_summary.csv", float_precision="round_trip")
    aggregate_detail = pd.read_csv(
        OUT / "noise_aggregate_repeats.csv",
        float_precision="round_trip",
    )
    aggregate_summary = pd.read_csv(
        OUT / "noise_aggregate_summary.csv",
        float_precision="round_trip",
    )
    expected_detail = (
        len(config["feature_sets"])
        * len(config["noise"]["multipliers"])
        * len(config["noise"]["seeds"])
        * len(MODES)
    )
    expected_summary = (
        len(config["feature_sets"])
        * len(config["noise"]["multipliers"])
        * len(MODES)
    )
    assert len(detail) == expected_detail
    assert len(summary) == expected_summary
    assert len(aggregate_detail) == (
        len(config["feature_sets"])
        * len(config["noise"]["multipliers"])
        * len(config["noise"]["seeds"])
    )
    assert len(aggregate_summary) == (
        len(config["feature_sets"]) * len(config["noise"]["multipliers"])
    )
    assert set(detail.seed) == set(config["noise"]["seeds"])
    assert set(detail.multiplier) == set(config["noise"]["multipliers"])
    for _, group in detail.groupby(["feature_set", "multiplier", "mode"]):
        assert len(group) == 5
        row = summary[
            (summary.feature_set == group.feature_set.iloc[0])
            & (summary.multiplier == group.multiplier.iloc[0])
            & (summary["mode"] == group["mode"].iloc[0])
        ].iloc[0]
        assert np.isclose(row.average_precision_mean, group.average_precision.mean())
        assert np.isclose(
            row.average_precision_std,
            group.average_precision.std(ddof=1),
        )
        assert np.isclose(row.f1_mean, group.f1.mean())
        assert np.isclose(row.normal_fpr_mean, group.normal_fpr.mean())
    for _, group in aggregate_detail.groupby(["feature_set", "multiplier"]):
        assert len(group) == 5
        row = aggregate_summary[
            (aggregate_summary.feature_set == group.feature_set.iloc[0])
            & (aggregate_summary.multiplier == group.multiplier.iloc[0])
        ].iloc[0]
        assert np.isclose(
            row.macro_average_precision_mean,
            group.macro_average_precision.mean(),
        )
        assert np.isclose(
            row.macro_average_precision_std,
            group.macro_average_precision.std(ddof=1),
        )
        assert np.isclose(row.macro_f1_mean, group.macro_f1.mean())
        assert np.isclose(row.any_mode_normal_fpr_mean, group.any_mode_normal_fpr.mean())
    test_metrics = pd.read_csv(OUT / "test_metrics.csv", float_precision="round_trip")
    test_aggregates = pd.read_csv(
        OUT / "test_aggregates.csv",
        float_precision="round_trip",
    )
    zero = summary[summary.multiplier == 0]
    for _, row in zero.iterrows():
        reference = test_metrics[
            (test_metrics.feature_set == row.feature_set)
            & (test_metrics["mode"] == row["mode"])
        ].iloc[0]
        assert np.isclose(row.average_precision_mean, reference.average_precision)
        assert np.isclose(row.f1_mean, reference.f1)
        assert np.isclose(row.normal_fpr_mean, reference.normal_fpr)
    for _, row in aggregate_summary[aggregate_summary.multiplier == 0].iterrows():
        reference = test_aggregates[test_aggregates.feature_set == row.feature_set].iloc[0]
        assert np.isclose(
            row.macro_average_precision_mean,
            reference.macro_average_precision,
        )
        assert np.isclose(row.macro_f1_mean, reference.macro_f1)
        assert np.isclose(
            row.any_mode_normal_fpr_mean,
            reference.any_mode_normal_fpr,
        )
    zero_detail = detail[detail.multiplier == 0]
    for _, group in zero_detail.groupby(["feature_set", "mode"]):
        assert group.average_precision.nunique() == 1
        assert group.f1.nunique() == 1
        assert group.normal_fpr.nunique() == 1

    binary_detail = pd.read_csv(
        OUT / "binary_noise_repeats.csv",
        float_precision="round_trip",
    )
    binary_summary = pd.read_csv(
        OUT / "binary_noise_summary.csv",
        float_precision="round_trip",
    )
    assert len(binary_detail) == len(config["noise"]["multipliers"]) * 5
    assert len(binary_summary) == len(config["noise"]["multipliers"])
    assert set(binary_detail.candidate) == {
        config["noise"]["binary_supplement"]["candidate"]
    }
    for _, group in binary_detail.groupby("multiplier"):
        assert len(group) == 5
    for _, row in binary_summary.iterrows():
        group = binary_detail[binary_detail.multiplier == row.multiplier]
        assert np.isclose(row.average_precision_mean, group.average_precision.mean())
        assert np.isclose(row.f1_mean, group.f1.mean())


def check_artifacts(manifest: dict) -> None:
    assert manifest["models_sha256"] == hashlib_sha256(MODEL_FILE)
    assert manifest["config_sha256"] == hashlib_sha256(CONFIG)
    required = [
        "candidate_mode_metrics.csv",
        "candidates.csv",
        "validation_predictions.csv",
        "validation_metrics.csv",
        "validation_aggregates.csv",
        "frozen_models.joblib",
        "freeze_manifest.json",
        "test_predictions.csv",
        "test_metrics.csv",
        "test_aggregates.csv",
        "mode_support.csv",
        "cases.csv",
        "mode_performance_heatmap.png",
        "mode_pr_curves.png",
        "noise_repeats.csv",
        "noise_summary.csv",
        "noise_aggregate_repeats.csv",
        "noise_aggregate_summary.csv",
        "noise_standard_deviations.csv",
        "noise_robustness.png",
        "noise_fpr.png",
        "binary_noise_repeats.csv",
        "binary_noise_summary.csv",
        "binary_noise_robustness.png",
    ]
    for name in required:
        path = OUT / name
        assert path.is_file() and path.stat().st_size > 0, name


def hashlib_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = read_config()
    manifest = verify_freeze()
    check_candidate_selection(config)
    check_thresholds(manifest, config)
    check_partition("validation", config)
    check_partition("test", config)
    check_aggregates()
    check_cases_and_support()
    check_noise(config)
    check_artifacts(manifest)
    print(
        "PASS: selection, thresholds, row labels, metrics, alarms, FPR denominators, "
        "aggregates, noise repeats, and artifacts verified"
    )


if __name__ == "__main__":
    main()
