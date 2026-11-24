"""Repeated 70/15/15 stratified split stability for D and C.

Run in the isolated Conda environment:
    conda run -n aiaa3111_corruption python -B -m Group19_D.repeated_splits

Seeds 42-46 reproduce the original split construction. Models are refit in
each split with the already-selected hyperparameters and thresholds are
recalibrated from validation only.
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
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from Group19_A.pipeline import MODES, TARGET, features, load_split, make_preprocessor
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
SPLIT_SEEDS = (42, 43, 44, 45, 46)
TARGET_FPR = 0.05
SOURCE_DATA = REPO / "Group19_A" / "data" / "raw" / "ai4i2020.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_metrics(y_mode, score, threshold, y_binary) -> dict:
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
    normal = y_binary == 0
    normal_fp = int(np.sum(alarm & normal))
    return {
        "average_precision": (
            float(average_precision_score(y_mode, score))
            if 0 < positives < len(y_mode)
            else np.nan
        ),
        "roc_auc": (
            float(roc_auc_score(y_mode, score))
            if 0 < positives < len(y_mode)
            else np.nan
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "support": positives,
        "normal_fp": normal_fp,
        "normal_denominator": int(normal.sum()),
        "normal_fpr": normal_fp / int(normal.sum()) if normal.any() else np.nan,
    }


def make_splits(df):
    splits = {}
    for seed in SPLIT_SEEDS:
        train, held = train_test_split(
            np.arange(len(df)),
            test_size=0.30,
            stratify=df[TARGET],
            random_state=seed,
        )
        validation, test = train_test_split(
            held,
            test_size=0.50,
            stratify=df.iloc[held][TARGET],
            random_state=seed,
        )
        splits[seed] = {
            "train": train,
            "validation": validation,
            "test": test,
        }
    frozen = load_split("train", engineered=False)
    frozen_index = frozen[0].index.to_numpy()
    if not np.array_equal(np.sort(splits[42]["train"]), np.sort(frozen_index)):
        raise AssertionError("Seed 42 no longer reproduces A's frozen training split")
    return splits


def subset(df, indices, engineered):
    part = df.iloc[indices]
    return features(part, engineered=engineered), part[TARGET], part[MODES]


def selected_c_model():
    validation = pd.read_csv(
        C_RESULTS / "validation_metrics.csv",
        float_precision="round_trip",
    )
    winner = validation.sort_values(
        "average_precision",
        ascending=False,
        kind="stable",
    ).iloc[0]
    pipeline = Pipeline([
        ("preprocess", make_preprocessor(engineered=True, scale=False)),
        ("model", make_model(
            "RandomForest",
            json.loads(winner.params),
            weighted=False,
            ratio=1.0,
            seed=42,
        )),
    ])
    return winner, pipeline


def aggregate_d(mode_rows, alarms, y_test, seed, feature_set):
    finite_ap = [
        row["average_precision"]
        for row in mode_rows
        if np.isfinite(row["average_precision"])
    ]
    finite_f1 = [
        row["f1"]
        for row in mode_rows
        if np.isfinite(row["f1"])
    ]
    normal = y_test.to_numpy() == 0
    union_alarm = alarms.any(axis=1)
    return {
        "seed": seed,
        "feature_set": feature_set,
        "evaluable_modes": len(finite_ap),
        "macro_average_precision": float(np.mean(finite_ap)),
        "macro_f1": float(np.mean(finite_f1)),
        "any_mode_normal_fp": int(union_alarm[normal].sum()),
        "any_mode_normal_denominator": int(normal.sum()),
        "any_mode_normal_fpr": float(union_alarm[normal].mean()),
    }


def summarize(detail, keys, metrics):
    rows = []
    for key, frame in detail.groupby(keys, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(keys, key))
        row["repeats"] = len(frame)
        for metric in metrics:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
            row[f"{metric}_min"] = float(frame[metric].min())
            row[f"{metric}_max"] = float(frame[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def run():
    config = read_config()
    verify_freeze()
    freeze = json.loads((OUT / "freeze_manifest.json").read_text(encoding="utf-8"))
    df = pd.read_csv(SOURCE_DATA)
    splits = make_splits(df)
    winner, _ = selected_c_model()
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

    split_rows = []
    d_mode_rows = []
    d_aggregate_rows = []
    c_rows = []
    d_prediction_frames = []
    c_prediction_frames = []

    for seed, indices in splits.items():
        split_rows.append({
            "seed": seed,
            **{
                f"{partition}_rows": len(part_indices)
                for partition, part_indices in indices.items()
            },
            **{
                f"{partition}_failures": int(df.iloc[part_indices][TARGET].sum())
                for partition, part_indices in indices.items()
            },
        })

        for feature_set in config["feature_sets"]:
            engineered = feature_set == "engineered"
            x_train, y_train, Y_train = subset(df, indices["train"], engineered)
            x_val, y_val, Y_val = subset(df, indices["validation"], engineered)
            x_test, y_test, Y_test = subset(df, indices["test"], engineered)
            preprocessor = make_preprocessor(engineered=engineered, scale=False)
            z_train = preprocessor.fit_transform(x_train)
            z_val = preprocessor.transform(x_val)
            z_test = preprocessor.transform(x_test)
            thresholds = {}
            alarms = np.zeros((len(x_test), len(MODES)), dtype=bool)
            mode_rows = []
            for mode_index, mode in enumerate(MODES):
                model = make_forest(
                    config,
                    freeze["selected"][feature_set]["params"],
                )
                model.fit(z_train, Y_train[mode].to_numpy())
                model.set_params(n_jobs=1)
                val_score = model.predict_proba(z_val)[:, 1]
                test_score = model.predict_proba(z_test)[:, 1]
                threshold, _ = threshold_for_fpr(
                    y_val,
                    val_score,
                    TARGET_FPR,
                )
                thresholds[mode] = threshold
                alarm = test_score >= threshold
                alarms[:, mode_index] = alarm
                metric = safe_metrics(
                    Y_test[mode],
                    test_score,
                    threshold,
                    y_test,
                )
                mode_rows.append(metric)
                d_mode_rows.append({
                    "seed": seed,
                    "feature_set": feature_set,
                    "mode": mode,
                    **metric,
                })
                d_prediction_frames.append(pd.DataFrame({
                    "seed": seed,
                    "feature_set": feature_set,
                    "mode": mode,
                    "row_index": x_test.index,
                    "y_mode": Y_test[mode].to_numpy(),
                    "y_binary": y_test.to_numpy(),
                    "score": test_score,
                    "threshold": threshold,
                    "alarm": alarm.astype(int),
                }))
            d_aggregate_rows.append(
                aggregate_d(mode_rows, alarms, y_test, seed, feature_set)
            )

        full_train = df.iloc[indices["train"]]
        full_val = df.iloc[indices["validation"]]
        full_test = df.iloc[indices["test"]]
        x_train = features(full_train, engineered=True)
        x_val = features(full_val, engineered=True)
        x_test = features(full_test, engineered=True)
        y_train = full_train[TARGET]
        y_val = full_val[TARGET]
        y_test = full_test[TARGET]
        estimator = Pipeline([
            ("preprocess", make_preprocessor(engineered=True, scale=False)),
            ("model", make_model(
                "RandomForest",
                json.loads(winner.params),
                weighted=False,
                ratio=1.0,
                seed=42,
            )),
        ])
        estimator.fit(x_train, y_train)
        val_score = estimator.predict_proba(x_val)[:, 1]
        threshold = threshold_max_f1(y_val, val_score)
        test_score = estimator.predict_proba(x_test)[:, 1]
        metric = safe_metrics(y_test, test_score, threshold, y_test)
        c_rows.append({
            "seed": seed,
            "candidate": winner.candidate,
            **metric,
        })
        c_prediction_frames.append(pd.DataFrame({
            "seed": seed,
            "candidate": winner.candidate,
            "row_index": x_test.index,
            "y_true": y_test.to_numpy(),
            "score": test_score,
            "threshold": threshold,
            "alarm": (test_score >= threshold).astype(int),
        }))
        print(f"completed split seed {seed}", flush=True)

    split_frame = pd.DataFrame(split_rows)
    d_modes = pd.DataFrame(d_mode_rows)
    d_aggregate = pd.DataFrame(d_aggregate_rows)
    c_metrics = pd.DataFrame(c_rows)
    split_frame.to_csv(OUT / "repeated_split_counts.csv", index=False)
    d_modes.to_csv(
        OUT / "repeated_split_d_mode_metrics.csv",
        index=False,
        float_format="%.17g",
    )
    d_aggregate.to_csv(
        OUT / "repeated_split_d_aggregate.csv",
        index=False,
        float_format="%.17g",
    )
    c_metrics.to_csv(
        OUT / "repeated_split_binary_metrics.csv",
        index=False,
        float_format="%.17g",
    )
    pd.concat(d_prediction_frames, ignore_index=True).to_csv(
        OUT / "repeated_split_d_predictions.csv",
        index=False,
        float_format="%.17g",
    )
    pd.concat(c_prediction_frames, ignore_index=True).to_csv(
        OUT / "repeated_split_binary_predictions.csv",
        index=False,
        float_format="%.17g",
    )
    d_summary = summarize(
        d_aggregate,
        ["feature_set"],
        ["macro_average_precision", "macro_f1", "any_mode_normal_fpr"],
    )
    c_summary = summarize(
        c_metrics,
        ["candidate"],
        ["average_precision", "f1", "recall", "normal_fpr"],
    )
    pd.concat([
        d_summary.assign(source="D"),
        c_summary.assign(source="C"),
    ], ignore_index=True).to_csv(
        OUT / "repeated_split_summary.csv",
        index=False,
        float_format="%.17g",
    )
    figures(d_aggregate, c_metrics)
    manifest = {
        "seeds": list(SPLIT_SEEDS),
        "split": "70/15/15 stratified by Machine failure",
        "d_target_fpr": TARGET_FPR,
        "c_candidate": winner.candidate,
        "source_sha256": {
            "raw_data": sha256(SOURCE_DATA),
            "D_models": sha256(MODEL_FILE),
            "C_models": sha256(C_RESULTS / "frozen_models.joblib"),
        },
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
    }
    (OUT / "repeated_split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("PASS: five-seed repeated split stability completed")


def figures(d_aggregate, c_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for axis, metric, title, scale in (
        (axes[0], "macro_average_precision", "Five-mode macro AP", 1.0),
        (axes[1], "any_mode_normal_fpr", "Any-mode normal FPR (%)", 100.0),
    ):
        for feature_set, color in (("raw", "#777777"), ("engineered", "#5B0F78")):
            values = d_aggregate[
                d_aggregate.feature_set == feature_set
            ][metric].to_numpy() * scale
            axis.plot(
                SPLIT_SEEDS,
                values,
                marker="o",
                linewidth=2,
                color=color,
                label=feature_set.title(),
            )
        axis.set_xticks(SPLIT_SEEDS)
        axis.set_xlabel("Split seed")
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_ylim(0, 0.85)
    axes[1].set_ylim(0, 35)
    fig.suptitle("Split stability for frozen D model settings")
    fig.savefig(OUT / "repeated_split_d_stability.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, metric, title, scale in (
        (axes[0], "average_precision", "C binary AP", 1.0),
        (axes[1], "f1", "C binary F1", 1.0),
    ):
        axis.plot(
            SPLIT_SEEDS,
            c_metrics[metric].to_numpy() * scale,
            marker="D",
            linewidth=2,
            color="#5B0F78",
        )
        axis.set_xticks(SPLIT_SEEDS)
        axis.set_xlabel("Split seed")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].set_ylim(0.7, 1.0)
    axes[1].set_ylim(0.65, 1.0)
    fig.suptitle("Split stability for C engineered RF")
    fig.savefig(OUT / "repeated_split_c_stability.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    run()
