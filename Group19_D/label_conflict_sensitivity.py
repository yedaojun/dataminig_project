"""Sensitivity analysis for the 27 audited total/mode-label conflicts.

Strategies:
1. retain: keep the original Machine failure target;
2. drop: remove conflict rows from train/validation/test and refit;
3. total_from_modes: define binary failure as any mode label being 1.

D's mode models keep their documented mode targets. They are refit only for
the drop strategy, where training rows change. C's validation-best binary
model is refit for every non-retain strategy with the same hyperparameters.

Run in the isolated Conda environment:
    conda run -n aiaa3111_corruption python -B -m Group19_D.label_conflict_sensitivity
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from Group19_A.pipeline import MODES, load_split, make_preprocessor
from Group19_C.run import make_model, threshold_max_f1
from Group19_D.run import (
    MODEL_FILE,
    OUT,
    make_forest,
    read_config,
    threshold_for_fpr,
    verify_freeze,
)

REPO = Path(__file__).resolve().parent.parent
C_RESULTS = REPO / "Group19_C" / "results"
STRATEGIES = ("retain", "drop_conflicts", "total_from_modes")
TARGET_FPR = 0.05


def safe_mode_metrics(y_mode, score, threshold, y_binary) -> dict:
    y_mode = np.asarray(y_mode, dtype=int)
    score = np.asarray(score, dtype=float)
    y_binary = np.asarray(y_binary, dtype=int)
    alarm = score >= float(threshold)
    tp = int(np.sum((y_mode == 1) & alarm))
    fp = int(np.sum((y_mode == 0) & alarm))
    tn = int(np.sum((y_mode == 0) & ~alarm))
    fn = int(np.sum((y_mode == 1) & ~alarm))
    positives = int(y_mode.sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else np.nan
    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator else np.nan
    ap = (
        float(average_precision_score(y_mode, score))
        if 0 < positives < len(y_mode)
        else np.nan
    )
    roc_auc = (
        float(roc_auc_score(y_mode, score))
        if 0 < positives < len(y_mode)
        else np.nan
    )
    normal = y_binary == 0
    normal_fp = int(np.sum(alarm & normal))
    return {
        "average_precision": ap,
        "roc_auc": roc_auc,
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "support": positives,
        "normal_fp": normal_fp,
        "normal_denominator": int(normal.sum()),
        "normal_fpr": normal_fp / int(normal.sum()),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_splits():
    data = {}
    for partition in ("train", "validation", "test"):
        x, y, Y = load_split(
            partition,
            engineered=True,
            allow_test=partition == "test",
        )
        conflict = y.ne(Y.max(axis=1))
        data[partition] = {
            "x_engineered": x,
            "y": y,
            "Y": Y,
            "conflict": conflict,
            "y_modes": Y.max(axis=1),
        }
    return data


def target_for_strategy(data, partition: str, strategy: str):
    frame = data[partition]
    keep = np.ones(len(frame["y"]), dtype=bool)
    if strategy == "retain":
        y_binary = frame["y"].copy()
    elif strategy == "drop_conflicts":
        keep = ~frame["conflict"].to_numpy()
        y_binary = frame["y"].copy()
    elif strategy == "total_from_modes":
        y_binary = frame["y_modes"].copy()
    else:
        raise ValueError(strategy)
    return keep, y_binary


def aggregate(mode_metrics_rows, alarms, y_binary):
    y_binary = np.asarray(y_binary, dtype=int)
    normal = y_binary == 0
    rows_by_mode = {
        mode: mode_metrics_rows[index]
        for index, mode in enumerate(MODES)
    }
    interpretable = [
        rows_by_mode[mode] for mode in MODES if mode != "RNF"
    ]
    macro_ap = float(np.nanmean([
        row["average_precision"] for row in mode_metrics_rows
    ]))
    macro_f1 = float(np.nanmean([
        row["f1"] for row in mode_metrics_rows
    ]))
    macro_ap_without_rnf = float(np.mean([
        row["average_precision"] for row in interpretable
    ]))
    macro_f1_without_rnf = float(np.mean([
        row["f1"] for row in interpretable
    ]))
    mode_fpr = float(np.mean([row["normal_fpr"] for row in mode_metrics_rows]))
    union_alarm = alarms.any(axis=1)
    return {
        "test_macro_average_precision_evaluable": macro_ap,
        "test_macro_average_precision_excluding_rnf": macro_ap_without_rnf,
        "test_macro_f1_evaluable": macro_f1,
        "test_macro_f1_excluding_rnf": macro_f1_without_rnf,
        "evaluable_modes": int(np.isfinite([
            row["average_precision"] for row in mode_metrics_rows
        ]).sum()),
        "test_mean_mode_fpr": mode_fpr,
        "test_any_mode_normal_fp": int(union_alarm[normal].sum()),
        "test_any_mode_normal_denominator": int(normal.sum()),
        "test_any_mode_normal_fpr": float(union_alarm[normal].mean()),
    }


def frozen_d_score(frozen, feature_set, x_engineered):
    engineered = feature_set == "engineered"
    x = x_engineered if engineered else x_engineered[list(x_engineered.columns[:6])]
    matrix = frozen[feature_set]["preprocessor"].transform(x)
    return {
        mode: frozen[feature_set]["models"][mode].predict_proba(matrix)[:, 1]
        for mode in MODES
    }


def d_strategies(data, config):
    verify_freeze()
    frozen = joblib.load(MODEL_FILE)
    freeze = json.loads((OUT / "freeze_manifest.json").read_text(encoding="utf-8"))
    predictions = []
    metrics_rows = []
    aggregate_rows = []
    support_rows = []

    for strategy in STRATEGIES:
        for feature_set in config["feature_sets"]:
            engineered = feature_set == "engineered"
            train_keep, y_train = target_for_strategy(data, "train", strategy)
            val_keep, y_val = target_for_strategy(data, "validation", strategy)
            test_keep, y_test = target_for_strategy(data, "test", strategy)
            x_train = (
                data["train"]["x_engineered"]
                if engineered
                else data["train"]["x_engineered"][list(data["train"]["x_engineered"].columns[:6])]
            )
            x_val = (
                data["validation"]["x_engineered"]
                if engineered
                else data["validation"]["x_engineered"][list(data["validation"]["x_engineered"].columns[:6])]
            )
            x_test = (
                data["test"]["x_engineered"]
                if engineered
                else data["test"]["x_engineered"][list(data["test"]["x_engineered"].columns[:6])]
            )
            Y_train = data["train"]["Y"]
            Y_val = data["validation"]["Y"]
            Y_test = data["test"]["Y"]

            if strategy == "drop_conflicts":
                x_train = x_train.iloc[train_keep]
                Y_train = Y_train.iloc[train_keep]
                x_val = x_val.iloc[val_keep]
                Y_val = Y_val.iloc[val_keep]
                y_val = y_val.iloc[val_keep]
                x_test = x_test.iloc[test_keep]
                Y_test = Y_test.iloc[test_keep]
                y_test = y_test.iloc[test_keep]
                preprocessor = make_preprocessor(engineered=engineered, scale=False)
                z_train = preprocessor.fit_transform(x_train)
                z_val = preprocessor.transform(x_val)
                z_test = preprocessor.transform(x_test)
                models = {}
                selected_params = freeze["selected"][feature_set]["params"]
                for mode in MODES:
                    model = make_forest(config, selected_params)
                    model.fit(z_train, Y_train[mode].to_numpy())
                    model.set_params(n_jobs=1)
                    models[mode] = model
            else:
                preprocessor = frozen[feature_set]["preprocessor"]
                models = frozen[feature_set]["models"]
                z_val = preprocessor.transform(x_val)
                z_test = preprocessor.transform(x_test)

            val_scores = {
                mode: models[mode].predict_proba(z_val)[:, 1]
                for mode in MODES
            }
            test_scores = {
                mode: models[mode].predict_proba(z_test)[:, 1]
                for mode in MODES
            }
            if strategy == "retain":
                thresholds = freeze["selected"][feature_set]["thresholds"]
            else:
                thresholds = {}
                for mode in MODES:
                    threshold, _ = threshold_for_fpr(
                        y_val,
                        val_scores[mode],
                        TARGET_FPR,
                    )
                    thresholds[mode] = threshold

            alarms = np.zeros((len(x_test), len(MODES)), dtype=bool)
            mode_rows = []
            for mode_index, mode in enumerate(MODES):
                alarm = test_scores[mode] >= thresholds[mode]
                alarms[:, mode_index] = alarm
                metric = safe_mode_metrics(
                    Y_test[mode],
                    test_scores[mode],
                    thresholds[mode],
                    y_test,
                )
                mode_rows.append(metric)
                metrics_rows.append({
                    "strategy": strategy,
                    "feature_set": feature_set,
                    "mode": mode,
                    "threshold": thresholds[mode],
                    **metric,
                })
                predictions.append(pd.DataFrame({
                    "strategy": strategy,
                    "feature_set": feature_set,
                    "mode": mode,
                    "row_index": x_test.index,
                    "y_mode": Y_test[mode].to_numpy(),
                    "y_binary": y_test.to_numpy(),
                    "score": test_scores[mode],
                    "threshold": thresholds[mode],
                    "alarm": alarm.astype(int),
                }))
            aggregate_rows.append({
                "strategy": strategy,
                "feature_set": feature_set,
                "train_rows": len(x_train),
                "validation_rows": len(x_val),
                "test_rows": len(x_test),
                "train_failures": int(y_train.sum()),
                "validation_failures": int(y_val.sum()),
                "test_failures": int(y_test.sum()),
                "test_normals": int((y_test == 0).sum()),
                **aggregate(mode_rows, alarms, y_test),
            })
            support_rows.append({
                "strategy": strategy,
                "partition": "test",
                "feature_set": feature_set,
                "rows": len(x_test),
                **{f"{mode}_support": int(Y_test[mode].sum()) for mode in MODES},
            })
    return (
        pd.DataFrame(metrics_rows),
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(support_rows),
        pd.concat(predictions, ignore_index=True),
    )


def c_strategy(data):
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    validation = pd.read_csv(
        C_RESULTS / "validation_metrics.csv",
        float_precision="round_trip",
    )
    winner = validation.sort_values(
        "average_precision",
        ascending=False,
        kind="stable",
    ).iloc[0]
    candidate = winner.candidate
    params = json.loads(winner.params)
    pipeline = Pipeline([
        ("preprocess", make_preprocessor(engineered=True, scale=False)),
        ("model", make_model(
            "RandomForest",
            params,
            weighted=False,
            ratio=1.0,
            seed=42,
        )),
    ])
    test_predictions = pd.read_csv(
        C_RESULTS / "test_predictions.csv",
        float_precision="round_trip",
    ).query("candidate == @candidate")
    rows = []
    prediction_frames = []
    for strategy in STRATEGIES:
        train_keep, y_train = target_for_strategy(data, "train", strategy)
        val_keep, y_val = target_for_strategy(data, "validation", strategy)
        test_keep, y_test = target_for_strategy(data, "test", strategy)
        x_train = data["train"]["x_engineered"].iloc[train_keep]
        x_val = data["validation"]["x_engineered"].iloc[val_keep]
        x_test = data["test"]["x_engineered"].iloc[test_keep]
        y_train = y_train.iloc[train_keep]
        y_val = y_val.iloc[val_keep]
        y_test = y_test.iloc[test_keep]

        if strategy == "retain":
            frame = test_predictions.set_index("row_index").loc[x_test.index]
            score = frame.score.to_numpy()
            threshold = float(winner.threshold)
        else:
            estimator = clone(pipeline)
            estimator.fit(x_train, y_train)
            val_score = estimator.predict_proba(x_val)[:, 1]
            threshold = threshold_max_f1(y_val, val_score)
            score = estimator.predict_proba(x_test)[:, 1]
        metric = safe_mode_metrics(y_test, score, threshold, y_test)
        rows.append({
            "strategy": strategy,
            "candidate": candidate,
            "model": winner.model,
            "feature_set": winner.feature_set,
            "train_rows": len(x_train),
            "validation_rows": len(x_val),
            "test_rows": len(x_test),
            "train_failures": int(y_train.sum()),
            "validation_failures": int(y_val.sum()),
            "test_failures": int(y_test.sum()),
            "threshold": threshold,
            **metric,
        })
        prediction_frames.append(pd.DataFrame({
            "strategy": strategy,
            "candidate": candidate,
            "row_index": x_test.index,
            "y_true": y_test.to_numpy(),
            "score": score,
            "threshold": threshold,
            "alarm": (score >= threshold).astype(int),
        }))
    return pd.DataFrame(rows), pd.concat(prediction_frames, ignore_index=True)


def figures(d_aggregate, c_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    positions = np.arange(len(STRATEGIES))
    for offset, feature_set, color in (
        (-0.18, "raw", "#777777"),
        (0.18, "engineered", "#5B0F78"),
    ):
        frame = d_aggregate[
            d_aggregate.feature_set == feature_set
        ].set_index("strategy").loc[list(STRATEGIES)]
        axes[0].bar(
            positions + offset,
            frame.test_macro_average_precision_excluding_rnf,
            width=0.34,
            color=color,
            label=feature_set.title(),
        )
        axes[1].bar(
            positions + offset,
            frame.test_any_mode_normal_fpr * 100,
            width=0.34,
            color=color,
            label=feature_set.title(),
        )
    axes[0].set_title("D macro AP (RNF excluded)")
    axes[0].set_ylabel("Average precision")
    axes[0].set_ylim(0, 0.8)
    axes[1].set_title("D any-mode normal FPR")
    axes[1].set_ylabel("False-positive rate (%)")
    axes[1].set_ylim(0, 25)
    for axis in axes:
        axis.set_xticks(positions, ["Retain", "Drop", "Modes define total"])
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    fig.suptitle("Label-conflict sensitivity: D diagnostic models")
    fig.savefig(OUT / "label_conflict_d_sensitivity.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    frame = c_metrics.set_index("strategy").loc[list(STRATEGIES)]
    width = 0.32
    axes[0].bar(
        positions - width / 2,
        frame.average_precision,
        width,
        color="#5B0F78",
        label="AP",
    )
    axes[0].bar(
        positions + width / 2,
        frame.f1,
        width,
        color="#B47CC7",
        label="F1",
    )
    axes[1].bar(
        positions - width / 2,
        frame.normal_fpr * 100,
        width,
        color="#C8A34A",
        label="Normal FPR",
    )
    axes[1].bar(
        positions + width / 2,
        frame.recall * 100,
        width,
        color="#5B0F78",
        label="Recall",
    )
    for axis in axes:
        axis.set_xticks(positions, ["Retain", "Drop", "Modes define total"])
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_ylim(0, 1.0)
    axes[1].set_ylim(0, 100)
    axes[0].set_ylabel("Test score")
    axes[1].set_ylabel("Percent")
    axes[0].set_title("C engineered RF: AP and F1")
    axes[1].set_title("C engineered RF: operating point")
    fig.suptitle("Binary-label sensitivity to conflict handling")
    fig.savefig(OUT / "label_conflict_binary_sensitivity.png", dpi=180)
    plt.close(fig)


def write_manifest():
    manifest = {
        "strategies": {
            "retain": "Use the original Machine failure target.",
            "drop_conflicts": "Remove all rows where Machine failure differs from any mode label.",
            "total_from_modes": "Define binary failure as any mode prediction being positive.",
        },
        "d_protocol": (
            "Frozen models for retain and total_from_modes; refit with the same "
            "selected hyperparameters for drop_conflicts."
        ),
        "c_protocol": (
            "Validation-best engineered RF; frozen for retain and refit with the "
            "same hyperparameters for the other strategies."
        ),
        "target_fpr": TARGET_FPR,
        "environment": {
            "python": platform.python_version(),
            **{
                package: importlib.metadata.version(package)
                for package in (
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "xgboost",
                    "joblib",
                )
            },
        },
        "source_sha256": {
            "D_models": sha256(MODEL_FILE),
            "C_models": sha256(C_RESULTS / "frozen_models.joblib"),
        },
    }
    (OUT / "label_conflict_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    config = read_config()
    data = load_splits()
    conflict_rows = []
    for partition in ("train", "validation", "test"):
        frame = data[partition]
        for row_index in frame["y"].index[frame["conflict"]]:
            conflict_rows.append({
                "partition": partition,
                "row_index": int(row_index),
                "total_failure": int(frame["y"].loc[row_index]),
                "any_mode": int(frame["y_modes"].loc[row_index]),
                **{
                    mode: int(frame["Y"].loc[row_index, mode])
                    for mode in MODES
                },
            })
    pd.DataFrame(conflict_rows).to_csv(
        OUT / "label_conflict_rows.csv",
        index=False,
    )
    d_metrics, d_aggregate, d_support, d_predictions = d_strategies(data, config)
    d_metrics.to_csv(
        OUT / "label_conflict_d_mode_metrics.csv",
        index=False,
        float_format="%.17g",
    )
    d_aggregate.to_csv(
        OUT / "label_conflict_d_aggregate.csv",
        index=False,
        float_format="%.17g",
    )
    d_support.to_csv(
        OUT / "label_conflict_d_support.csv",
        index=False,
    )
    d_predictions.to_csv(
        OUT / "label_conflict_d_predictions.csv",
        index=False,
        float_format="%.17g",
    )
    c_metrics, c_predictions = c_strategy(data)
    c_metrics.to_csv(
        OUT / "label_conflict_binary_metrics.csv",
        index=False,
        float_format="%.17g",
    )
    c_predictions.to_csv(
        OUT / "label_conflict_binary_predictions.csv",
        index=False,
        float_format="%.17g",
    )
    figures(d_aggregate, c_metrics)
    write_manifest()
    print(
        "PASS: label-conflict sensitivity completed for D diagnostics and C binary model"
    )


if __name__ == "__main__":
    main()
