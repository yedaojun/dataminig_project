"""Workstream D: five-mode diagnosis, frozen thresholds, and test evaluation.

Run from the repository root:
    python -m Group19_D.run tune
    python -m Group19_D.run evaluate-test
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from Group19_A.pipeline import (
    INPUTS,
    MODES,
    TARGET,
    fingerprint,
    load_split,
    make_preprocessor,
    split_fingerprint,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CONFIG = HERE / "config.json"
MODEL_FILE = OUT / "frozen_models.joblib"
FREEZE_FILE = OUT / "freeze_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_config() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["seed"] != 42:
        raise ValueError("D requires the shared seed 42")
    if config["feature_sets"] != ["raw", "engineered"]:
        raise ValueError("D requires raw and engineered feature sets in that order")
    if config["selection"]["metric"] != "macro_average_precision_across_five_modes":
        raise ValueError("Unexpected D selection metric")
    target = float(config["diagnostic_threshold"]["target_false_positive_rate"])
    if not np.isclose(target, 0.05):
        raise ValueError("D requires the prespecified 5% validation-normal FPR target")
    return config


def check_rows(x, y, Y, partition: str) -> None:
    if not x.index.is_unique or not x.index.equals(y.index) or not x.index.equals(Y.index):
        raise ValueError(f"{partition}: feature, binary-label, and mode-label rows differ")
    if not y.isin([0, 1]).all() or not Y.isin([0, 1]).all().all():
        raise ValueError(f"{partition}: labels must be binary")
    forbidden = set(["UDI", "Product ID", TARGET] + MODES).intersection(x.columns)
    if forbidden:
        raise ValueError(f"{partition}: forbidden input labels: {sorted(forbidden)}")
    if list(x.columns[: len(INPUTS)]) != INPUTS:
        raise ValueError(f"{partition}: unexpected or reordered input columns")


def threshold_for_fpr(y_total, scores, target_fpr: float):
    """Lowest exact threshold allowing at most floor(target*n) normal alarms.

    The threshold is placed just above the first excluded normal score. This
    makes `score >= threshold` explicit and guarantees the finite-sample budget
    even when validation scores contain ties.
    """
    y_total = np.asarray(y_total)
    scores = np.asarray(scores)
    normal_scores = np.sort(scores[y_total == 0])[::-1]
    if len(normal_scores) == 0:
        raise ValueError("No total-failure-normal validation rows")
    allowed = int(np.floor(target_fpr * len(normal_scores)))
    if allowed >= len(normal_scores):
        return float("-inf"), allowed
    threshold = float(np.nextafter(normal_scores[allowed], np.inf))
    actual = int(np.sum(scores[y_total == 0] >= threshold))
    if actual > allowed:
        raise AssertionError("Threshold exceeds the finite-sample FPR budget")
    return threshold, allowed


def mode_metrics(y_mode, scores, threshold, y_total) -> dict:
    y_mode = np.asarray(y_mode)
    scores = np.asarray(scores)
    alarm = scores >= threshold
    tn, fp, fn, tp = map(int, confusion_matrix(y_mode, alarm, labels=[0, 1]).ravel())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    normal = np.asarray(y_total) == 0
    normal_fp = int(np.sum(alarm & normal))
    mode_negative = y_mode == 0
    mode_negative_fp = int(np.sum(alarm & mode_negative))
    return {
        "average_precision": float(average_precision_score(y_mode, scores)),
        "roc_auc": float(roc_auc_score(y_mode, scores)),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "support": int(y_mode.sum()),
        "normal_fp": normal_fp,
        "normal_denominator": int(normal.sum()),
        "normal_fpr": normal_fp / int(normal.sum()),
        "mode_negative_fp": mode_negative_fp,
        "mode_negative_denominator": int(mode_negative.sum()),
        "mode_negative_fpr": mode_negative_fp / int(mode_negative.sum()),
    }


def aggregate_metrics(predictions: pd.DataFrame) -> dict:
    per_mode = []
    for mode, group in predictions.groupby("mode", sort=False):
        per_mode.append(mode_metrics(group.y_true, group.score, group.threshold.iloc[0],
                                     group.total_failure))
    tp = sum(row["tp"] for row in per_mode)
    fp = sum(row["fp"] for row in per_mode)
    fn = sum(row["fn"] for row in per_mode)
    micro_precision = tp / (tp + fp) if tp + fp else 0.0
    micro_recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    normal = predictions.total_failure.to_numpy() == 0
    any_alarm = (
        predictions.assign(alarm=predictions.score >= predictions.threshold)
        .groupby("row_index")
        .alarm
        .max()
        .to_numpy()
    )
    row_order = predictions.drop_duplicates("row_index").sort_values("row_index")
    normal_by_row = row_order.total_failure.to_numpy() == 0
    any_normal_alarm = any_alarm[normal_by_row]
    return {
        "macro_average_precision": float(np.mean([row["average_precision"] for row in per_mode])),
        "macro_roc_auc": float(np.mean([row["roc_auc"] for row in per_mode])),
        "macro_precision": float(np.mean([row["precision"] for row in per_mode])),
        "macro_recall": float(np.mean([row["recall"] for row in per_mode])),
        "macro_f1": float(np.mean([row["f1"] for row in per_mode])),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "micro_tp": tp,
        "micro_fp": fp,
        "micro_fn": fn,
        "any_mode_normal_fp": int(any_normal_alarm.sum()),
        "any_mode_normal_denominator": int(normal_by_row.sum()),
        "any_mode_normal_fpr": float(any_normal_alarm.mean()),
    }


def make_forest(config: dict, candidate: dict) -> RandomForestClassifier:
    params = dict(config["fixed_forest_params"])
    params.update(candidate)
    return RandomForestClassifier(
        **params,
        class_weight=None,
        random_state=config["seed"],
    )


def prediction_rows(partition, row_index, y_total, Y, scores_by_mode,
                    thresholds, feature_set, candidate):
    frames = []
    for mode in MODES:
        score = np.asarray(scores_by_mode[mode])
        threshold = float(thresholds[mode])
        frames.append(pd.DataFrame({
            "partition": partition,
            "row_index": np.asarray(row_index),
            "candidate": candidate,
            "model": "RandomForest",
            "feature_set": feature_set,
            "mode": mode,
            "y_true": np.asarray(Y[mode]),
            "total_failure": np.asarray(y_total),
            "score": score,
            "threshold": threshold,
            "alarm": (score >= threshold).astype(int),
        }))
    return pd.concat(frames, ignore_index=True)


def tune() -> None:
    config = read_config()
    OUT.mkdir(exist_ok=True)
    candidate_rows = []
    selected_predictions = {}
    validation_metrics = []
    selected_models = {}
    selected_records = {}

    for feature_set in config["feature_sets"]:
        engineered = feature_set == "engineered"
        x_train, y_train, Y_train = load_split("train", engineered=engineered)
        x_val, y_val, Y_val = load_split("validation", engineered=engineered)
        check_rows(x_train, y_train, Y_train, "train")
        check_rows(x_val, y_val, Y_val, "validation")

        preprocessor = make_preprocessor(engineered=engineered, scale=False)
        z_train = preprocessor.fit_transform(x_train)
        z_val = preprocessor.transform(x_val)
        if z_train.shape[1] != (11 if engineered else 8):
            raise AssertionError("Unexpected preprocessed width")
        if not np.isfinite(z_train).all() or not np.isfinite(z_val).all():
            raise AssertionError("Non-finite preprocessed values")

        candidate_results = []
        for ordinal, candidate in enumerate(config["candidates"]):
            candidate_id = f"{feature_set}_rf_{ordinal}"
            models = {}
            scores = {}
            aps = []
            aucs = []
            for mode in MODES:
                model = make_forest(config, candidate)
                model.fit(z_train, Y_train[mode].to_numpy())
                # Predict serially so tree-probability summation is deterministic
                # after serialization and reload.
                model.set_params(n_jobs=1)
                score = model.predict_proba(z_val)[:, 1]
                models[mode] = model
                scores[mode] = score
                ap = float(average_precision_score(Y_val[mode], score))
                auc = float(roc_auc_score(Y_val[mode], score))
                aps.append(ap)
                aucs.append(auc)
                candidate_rows.append({
                    "candidate": candidate_id,
                    "model": "RandomForest",
                    "feature_set": feature_set,
                    "candidate_order": ordinal,
                    "params": json.dumps(candidate, sort_keys=True),
                    "mode": mode,
                    "validation_average_precision": ap,
                    "validation_roc_auc": auc,
                    "validation_support": int(Y_val[mode].sum()),
                })
            macro_ap = float(np.mean(aps))
            macro_auc = float(np.mean(aucs))
            candidate_results.append({
                "candidate": candidate_id,
                "candidate_order": ordinal,
                "params": candidate,
                "models": models,
                "scores": scores,
                "macro_ap": macro_ap,
                "macro_auc": macro_auc,
            })
            print(f"{candidate_id}: validation macro AP={macro_ap:.6f}", flush=True)

        best = candidate_results[0]
        for result in candidate_results[1:]:
            if result["macro_ap"] > best["macro_ap"]:
                best = result

        thresholds = {}
        allowed_by_mode = {}
        for mode in MODES:
            threshold, allowed = threshold_for_fpr(
                y_val,
                best["scores"][mode],
                config["diagnostic_threshold"]["target_false_positive_rate"],
            )
            thresholds[mode] = threshold
            allowed_by_mode[mode] = allowed

        frame = prediction_rows(
            "validation",
            y_val.index,
            y_val,
            Y_val,
            best["scores"],
            thresholds,
            feature_set,
            best["candidate"],
        )
        selected_predictions[feature_set] = frame
        for mode in MODES:
            metrics = mode_metrics(
                Y_val[mode],
                best["scores"][mode],
                thresholds[mode],
                y_val,
            )
            validation_metrics.append({
                "partition": "validation",
                "candidate": best["candidate"],
                "model": "RandomForest",
                "feature_set": feature_set,
                "mode": mode,
                "params": json.dumps(best["params"], sort_keys=True),
                "validation_normal_fpr_target": config["diagnostic_threshold"][
                    "target_false_positive_rate"
                ],
                "allowed_validation_normal_fp": allowed_by_mode[mode],
                **metrics,
            })

        selected_models[feature_set] = {
            "preprocessor": preprocessor,
            "models": best["models"],
            "candidate": best["candidate"],
            "feature_set": feature_set,
            "engineered": engineered,
            "input_columns": list(x_train.columns),
            "feature_names": list(preprocessor.get_feature_names_out()),
        }
        selected_records[feature_set] = {
            "candidate": best["candidate"],
            "candidate_order": best["candidate_order"],
            "params": best["params"],
            "engineered": engineered,
            "input_columns": list(x_train.columns),
            "feature_names": list(preprocessor.get_feature_names_out()),
            "validation_macro_average_precision": best["macro_ap"],
            "validation_macro_roc_auc": best["macro_auc"],
            "validation_normal_denominator": int((y_val == 0).sum()),
            "thresholds": thresholds,
            "allowed_validation_normal_fp": allowed_by_mode,
        }

    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(OUT / "candidate_mode_metrics.csv", index=False, float_format="%.17g")
    candidate_summary = candidates.groupby(
        ["candidate", "model", "feature_set", "candidate_order", "params"],
        as_index=False,
    ).agg(
        validation_macro_average_precision=("validation_average_precision", "mean"),
        validation_macro_roc_auc=("validation_roc_auc", "mean"),
        modes=("mode", "count"),
    )
    candidate_summary["selected"] = [
        int(row.candidate == selected_records[row.feature_set]["candidate"])
        for row in candidate_summary.itertuples()
    ]
    candidate_summary.to_csv(OUT / "candidates.csv", index=False, float_format="%.17g")

    validation_pred = pd.concat(selected_predictions.values(), ignore_index=True)
    validation_pred.to_csv(
        OUT / "validation_predictions.csv",
        index=False,
        float_format="%.17g",
    )
    validation_metrics_frame = pd.DataFrame(validation_metrics)
    validation_metrics_frame.to_csv(
        OUT / "validation_metrics.csv",
        index=False,
        float_format="%.17g",
    )
    aggregate_rows = []
    for feature_set in config["feature_sets"]:
        aggregate_rows.append({
            "partition": "validation",
            "feature_set": feature_set,
            "candidate": selected_records[feature_set]["candidate"],
            **aggregate_metrics(selected_predictions[feature_set]),
        })
    pd.DataFrame(aggregate_rows).to_csv(
        OUT / "validation_aggregates.csv",
        index=False,
        float_format="%.17g",
    )

    joblib.dump(selected_models, MODEL_FILE, compress=3)
    restored = joblib.load(MODEL_FILE)
    for feature_set in config["feature_sets"]:
        x_val, y_val, Y_val = load_split(
            "validation",
            engineered=feature_set == "engineered",
        )
        z_val = restored[feature_set]["preprocessor"].transform(x_val)
        for mode in MODES:
            replay = restored[feature_set]["models"][mode].predict_proba(z_val)[:, 1]
            original = (
                validation_pred[
                    (validation_pred.feature_set == feature_set)
                    & (validation_pred["mode"] == mode)
                ]
                .sort_values("row_index")
                .score
                .to_numpy()
            )
            row_order = np.argsort(x_val.index.to_numpy(), kind="mergesort")
            if not np.allclose(replay[row_order], original, rtol=0, atol=1e-15):
                raise AssertionError(f"Frozen validation replay differs: {feature_set}/{mode}")

    manifest = {
        "status": "frozen_before_test",
        "protocol_version": config["protocol_version"],
        "data_sha256": fingerprint(),
        "indices_sha256": split_fingerprint(),
        "config_sha256": sha256(CONFIG),
        "run_code_sha256": sha256(Path(__file__)),
        "pipeline_code_sha256": sha256(HERE.parent / "Group19_A" / "pipeline.py"),
        "models_sha256": sha256(MODEL_FILE),
        "seed": config["seed"],
        "model": config["model"],
        "score_direction": "higher_means_more_likely_mode",
        "replay_tolerance": {"rtol": 0, "atol": 1e-15},
        "selection": config["selection"],
        "diagnostic_threshold": config["diagnostic_threshold"],
        "environment": {
            "python": platform.python_version(),
            **{
                package: importlib.metadata.version(package)
                for package in ("numpy", "pandas", "scikit-learn", "joblib")
            },
        },
        "sklearn_version": sklearn.__version__,
        "selected": selected_records,
    }
    FREEZE_FILE.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "PASS: selected and froze D models and per-mode thresholds; "
        "test partition not accessed"
    )


def verify_freeze() -> dict:
    manifest = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    checks = (
        ("data_sha256", fingerprint()),
        ("indices_sha256", split_fingerprint()),
        ("config_sha256", sha256(CONFIG)),
        ("run_code_sha256", sha256(Path(__file__))),
        ("pipeline_code_sha256", sha256(HERE.parent / "Group19_A" / "pipeline.py")),
        ("models_sha256", sha256(MODEL_FILE)),
    )
    for field, actual in checks:
        if manifest[field] != actual:
            raise ValueError(f"Frozen {field} changed; rerun tune before test evaluation")
    if manifest["status"] != "frozen_before_test":
        raise ValueError("Unexpected D freeze status")
    return manifest


def evaluate_test() -> None:
    config = read_config()
    manifest = verify_freeze()
    frozen = joblib.load(MODEL_FILE)
    all_predictions = []
    test_metrics = []
    support_rows = []
    raw_test_by_set = {}

    for feature_set in config["feature_sets"]:
        engineered = feature_set == "engineered"
        x_train, _, Y_train = load_split("train", engineered=engineered)
        x_val, _, Y_val = load_split("validation", engineered=engineered)
        x_test, y_test, Y_test = load_split(
            "test",
            engineered=engineered,
            allow_test=True,
        )
        check_rows(x_test, y_test, Y_test, "test")
        raw_test_by_set[feature_set] = x_test
        for mode in MODES:
            support_rows.extend([
                {"partition": "train", "mode": mode, "support": int(Y_train[mode].sum())},
                {"partition": "validation", "mode": mode, "support": int(Y_val[mode].sum())},
                {"partition": "test", "mode": mode, "support": int(Y_test[mode].sum())},
            ])
        support_rows.extend([
            {"partition": "train", "mode": "TOTAL_FAILURE", "support": 237},
            {"partition": "validation", "mode": "TOTAL_FAILURE", "support": 51},
            {"partition": "test", "mode": "TOTAL_FAILURE", "support": int(y_test.sum())},
        ])

        record = frozen[feature_set]
        z_test = record["preprocessor"].transform(x_test)
        scores = {
            mode: record["models"][mode].predict_proba(z_test)[:, 1]
            for mode in MODES
        }
        thresholds = manifest["selected"][feature_set]["thresholds"]
        frame = prediction_rows(
            "test",
            y_test.index,
            y_test,
            Y_test,
            scores,
            thresholds,
            feature_set,
            record["candidate"],
        )
        all_predictions.append(frame)
        for mode in MODES:
            test_metrics.append({
                "partition": "test",
                "candidate": record["candidate"],
                "model": "RandomForest",
                "feature_set": feature_set,
                "mode": mode,
                "params": json.dumps(
                    manifest["selected"][feature_set]["params"],
                    sort_keys=True,
                ),
                **mode_metrics(Y_test[mode], scores[mode], thresholds[mode], y_test),
            })

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(
        OUT / "test_predictions.csv",
        index=False,
        float_format="%.17g",
    )
    metrics_frame = pd.DataFrame(test_metrics)
    metrics_frame.to_csv(
        OUT / "test_metrics.csv",
        index=False,
        float_format="%.17g",
    )
    aggregates = []
    for feature_set in config["feature_sets"]:
        aggregates.append({
            "partition": "test",
            "feature_set": feature_set,
            "candidate": frozen[feature_set]["candidate"],
            **aggregate_metrics(predictions[predictions.feature_set == feature_set]),
        })
    pd.DataFrame(aggregates).to_csv(
        OUT / "test_aggregates.csv",
        index=False,
        float_format="%.17g",
    )
    support_frame = pd.DataFrame(support_rows).drop_duplicates()
    support_frame.to_csv(OUT / "mode_support.csv", index=False)

    case_rows = []
    for feature_set in config["feature_sets"]:
        x_test = raw_test_by_set[feature_set]
        for mode in MODES:
            group = predictions[
                (predictions.feature_set == feature_set)
                & (predictions["mode"] == mode)
            ].copy()
            for case_type, mask in (
                ("detected", (group.y_true == 1) & (group.alarm == 1)),
                ("false_positive", (group.y_true == 0) & (group.alarm == 1)),
                ("missed", (group.y_true == 1) & (group.alarm == 0)),
            ):
                candidates = group[mask].sort_values("score", ascending=False)
                if candidates.empty:
                    continue
                row = candidates.iloc[0]
                details = {
                    "feature_set": feature_set,
                    "mode": mode,
                    "case_type": case_type,
                    "row_index": int(row.row_index),
                    "y_true": int(row.y_true),
                    "total_failure": int(row.total_failure),
                    "score": float(row.score),
                    "threshold": float(row.threshold),
                    "alarm": int(row.alarm),
                }
                for column in INPUTS:
                    details[column] = x_test.loc[int(row.row_index), column]
                case_rows.append(details)
    pd.DataFrame(case_rows).to_csv(
        OUT / "cases.csv",
        index=False,
        float_format="%.17g",
    )

    plot_mode_heatmap(metrics_frame)
    plot_mode_pr_curves(predictions)
    print("PASS: evaluated frozen D models on A's test split")


def plot_mode_heatmap(metrics: pd.DataFrame) -> None:
    modes = list(MODES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "average_precision", "Test average precision"),
        (axes[1], "f1", "Test F1 at validation 5% FPR threshold"),
    ):
        matrix = np.empty((len(modes), 2))
        annotations = []
        for row_index, mode in enumerate(modes):
            support = int(metrics[metrics["mode"] == mode].support.iloc[0])
            labels = []
            for column_index, feature_set in enumerate(("raw", "engineered")):
                value = float(
                    metrics[
                        (metrics["mode"] == mode)
                        & (metrics.feature_set == feature_set)
                    ][metric].iloc[0]
                )
                matrix[row_index, column_index] = value
                labels.append(f"{value:.3f}")
            annotations.append(labels)
        image = axis.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap="YlGnBu")
        axis.set_xticks([0, 1], ["raw", "engineered"])
        axis.set_yticks(
            range(len(modes)),
            [f"{mode} (n={int(metrics[metrics['mode'] == mode].support.iloc[0])})"
             for mode in modes],
        )
        for row_index in range(len(modes)):
            for column_index in range(2):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    annotations[row_index][column_index],
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "black",
                    fontsize=10,
                )
        axis.set_title(title)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Group 19 D: test performance by failure mode and input set")
    fig.savefig(OUT / "mode_performance_heatmap.png", dpi=180)
    plt.close(fig)


def plot_mode_pr_curves(predictions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, len(MODES), figsize=(18, 7), constrained_layout=True)
    for row_index, feature_set in enumerate(("raw", "engineered")):
        for column_index, mode in enumerate(MODES):
            axis = axes[row_index, column_index]
            group = predictions[
                (predictions.feature_set == feature_set)
                & (predictions["mode"] == mode)
            ]
            precision, recall, _ = precision_recall_curve(group.y_true, group.score)
            ap = average_precision_score(group.y_true, group.score)
            axis.plot(recall, precision, color="#2878b5")
            axis.axhline(
                float(group.y_true.mean()),
                color="grey",
                linestyle="--",
                linewidth=1,
            )
            axis.set_title(
                f"{feature_set} / {mode}\nAP={ap:.3f}, n={int(group.y_true.sum())}",
                fontsize=9,
            )
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1.03)
            axis.grid(alpha=0.2)
            if column_index == 0:
                axis.set_ylabel("Precision")
            if row_index == len(("raw", "engineered")) - 1:
                axis.set_xlabel("Recall")
    fig.suptitle("Group 19 D: test precision-recall curves")
    fig.savefig(OUT / "mode_pr_curves.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["tune", "evaluate-test"])
    command = parser.parse_args().command
    {"tune": tune, "evaluate-test": evaluate_test}[command]()


if __name__ == "__main__":
    main()
