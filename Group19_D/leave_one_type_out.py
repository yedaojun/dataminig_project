"""Leave-one-product-type-out generalization for D and C.

Protocol:
- train on A's training rows whose Type is not the held-out type;
- calibrate on A's validation rows of the observed types;
- evaluate on A's test rows whose Type equals the held-out type.

The in-domain reference uses the original all-type frozen models on the same
held-out test rows, allowing a direct comparison.

Run in the isolated Conda environment:
    conda run -n aiaa3111_corruption python -B -m Group19_D.leave_one_type_out
"""
from __future__ import annotations

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
TYPES = ("L", "M", "H")
TARGET_FPR = 0.05


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


def load_data():
    data = {}
    for partition in ("train", "validation", "test"):
        x, y, Y = load_split(
            partition,
            engineered=True,
            allow_test=partition == "test",
        )
        data[partition] = {"x": x, "y": y, "Y": Y}
    return data


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
    return winner, pipeline


def d_model_setup(config, freeze):
    settings = {}
    for feature_set in config["feature_sets"]:
        settings[feature_set] = {
            "engineered": feature_set == "engineered",
            "params": freeze["selected"][feature_set]["params"],
        }
    return settings


def evaluate_d_scores(feature_set, scores, thresholds, x_subset, y_subset, Y_subset, evaluation):
    rows = []
    alarms = np.zeros((len(x_subset), len(MODES)), dtype=bool)
    for mode_index, mode in enumerate(MODES):
        alarm = scores[mode] >= thresholds[mode]
        alarms[:, mode_index] = alarm
        rows.append({
            "evaluation": evaluation,
            "source": "D",
            "feature_set": feature_set,
            "mode": mode,
            **safe_metrics(
                Y_subset[mode],
                scores[mode],
                thresholds[mode],
                y_subset,
            ),
        })
    mode_rows = rows
    finite_ap = [
        row["average_precision"]
        for row in mode_rows
        if np.isfinite(row["average_precision"])
    ]
    finite_f1 = [
        row["f1"] for row in mode_rows if np.isfinite(row["f1"])
    ]
    normal = y_subset.to_numpy() == 0
    union_alarm = alarms.any(axis=1)
    aggregate = {
        "evaluation": evaluation,
        "source": "D",
        "feature_set": feature_set,
        "evaluable_modes": len(finite_ap),
        "macro_average_precision": float(np.mean(finite_ap)) if finite_ap else np.nan,
        "macro_f1": float(np.mean(finite_f1)) if finite_f1 else np.nan,
        "any_mode_normal_fp": int(union_alarm[normal].sum()),
        "any_mode_normal_denominator": int(normal.sum()),
        "any_mode_normal_fpr": float(union_alarm[normal].mean()),
    }
    return mode_rows, aggregate


def run_d(data, config):
    verify_freeze()
    freeze = json.loads((OUT / "freeze_manifest.json").read_text(encoding="utf-8"))
    frozen = joblib.load(MODEL_FILE)
    settings = d_model_setup(config, freeze)
    metric_rows = []
    aggregate_rows = []
    support_rows = []
    prediction_frames = []

    for heldout in TYPES:
        train_mask = data["train"]["x"].Type.ne(heldout).to_numpy()
        val_mask = data["validation"]["x"].Type.ne(heldout).to_numpy()
        test_mask = data["test"]["x"].Type.eq(heldout).to_numpy()
        support_rows.append({
            "heldout_type": heldout,
            "train_rows": int(train_mask.sum()),
            "validation_rows": int(val_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "test_failures": int(data["test"]["y"].iloc[test_mask].sum()),
            **{
                f"{mode}_support": int(
                    data["test"]["Y"][mode].iloc[test_mask].sum()
                )
                for mode in MODES
            },
        })

        for feature_set, setting in settings.items():
            engineered = setting["engineered"]
            x_train_full = data["train"]["x"]
            x_val_full = data["validation"]["x"]
            x_test_full = data["test"]["x"]
            if not engineered:
                columns = list(x_train_full.columns[:6])
                x_train_full = x_train_full[columns]
                x_val_full = x_val_full[columns]
                x_test_full = x_test_full[columns]
            x_train = x_train_full.iloc[train_mask]
            x_val = x_val_full.iloc[val_mask]
            x_test = x_test_full.iloc[test_mask]
            y_train = data["train"]["y"].iloc[train_mask]
            y_val = data["validation"]["y"].iloc[val_mask]
            y_test = data["test"]["y"].iloc[test_mask]
            Y_train = data["train"]["Y"].iloc[train_mask]
            Y_test = data["test"]["Y"].iloc[test_mask]

            preprocessor = make_preprocessor(engineered=engineered, scale=False)
            z_train = preprocessor.fit_transform(x_train)
            z_val = preprocessor.transform(x_val)
            z_test = preprocessor.transform(x_test)
            models = {}
            val_scores = {}
            test_scores = {}
            thresholds = {}
            for mode in MODES:
                model = make_forest(config, setting["params"])
                model.fit(z_train, Y_train[mode].to_numpy())
                model.set_params(n_jobs=1)
                models[mode] = model
                val_scores[mode] = model.predict_proba(z_val)[:, 1]
                test_scores[mode] = model.predict_proba(z_test)[:, 1]
                threshold, _ = threshold_for_fpr(
                    y_val,
                    val_scores[mode],
                    TARGET_FPR,
                )
                thresholds[mode] = threshold

            mode_rows, aggregate = evaluate_d_scores(
                feature_set,
                test_scores,
                thresholds,
                x_test,
                y_test,
                Y_test,
                "loto",
            )
            for row in mode_rows:
                row["heldout_type"] = heldout
                metric_rows.append(row)
            aggregate["heldout_type"] = heldout
            aggregate_rows.append(aggregate)
            for mode in MODES:
                prediction_frames.append(pd.DataFrame({
                    "evaluation": "loto",
                    "feature_set": feature_set,
                    "heldout_type": heldout,
                    "mode": mode,
                    "row_index": x_test.index,
                    "y_mode": Y_test[mode].to_numpy(),
                    "y_binary": y_test.to_numpy(),
                    "score": test_scores[mode],
                    "threshold": thresholds[mode],
                    "alarm": (test_scores[mode] >= thresholds[mode]).astype(int),
                }))

            # Original all-type reference evaluated on the same held-out rows.
            ref_record = frozen[feature_set]
            ref_matrix = ref_record["preprocessor"].transform(
                x_test_full.iloc[test_mask]
            )
            ref_scores = {
                mode: ref_record["models"][mode].predict_proba(ref_matrix)[:, 1]
                for mode in MODES
            }
            ref_thresholds = freeze["selected"][feature_set]["thresholds"]
            ref_mode_rows, ref_aggregate = evaluate_d_scores(
                feature_set,
                ref_scores,
                ref_thresholds,
                x_test,
                y_test,
                Y_test,
                "in_domain_reference",
            )
            for row in ref_mode_rows:
                row["heldout_type"] = heldout
                metric_rows.append(row)
            ref_aggregate["heldout_type"] = heldout
            aggregate_rows.append(ref_aggregate)

    return pd.DataFrame(metric_rows), pd.DataFrame(aggregate_rows), pd.DataFrame(support_rows), pd.concat(prediction_frames, ignore_index=True)


def run_c(data):
    winner, pipeline = selected_c_model()
    all_test_pred = pd.read_csv(
        C_RESULTS / "test_predictions.csv",
        float_precision="round_trip",
    ).query("candidate == @winner.candidate")
    rows = []
    prediction_frames = []
    for heldout in TYPES:
        train_mask = data["train"]["x"].Type.ne(heldout).to_numpy()
        val_mask = data["validation"]["x"].Type.ne(heldout).to_numpy()
        test_mask = data["test"]["x"].Type.eq(heldout).to_numpy()
        x_train = data["train"]["x"].iloc[train_mask]
        x_val = data["validation"]["x"].iloc[val_mask]
        x_test = data["test"]["x"].iloc[test_mask]
        y_train = data["train"]["y"].iloc[train_mask]
        y_val = data["validation"]["y"].iloc[val_mask]
        y_test = data["test"]["y"].iloc[test_mask]

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
        loto_score = estimator.predict_proba(x_test)[:, 1]
        loto_metric = safe_metrics(y_test, loto_score, threshold, y_test)
        rows.append({
            "evaluation": "loto",
            "source": "C",
            "heldout_type": heldout,
            "candidate": winner.candidate,
            "threshold": threshold,
            **loto_metric,
        })
        prediction_frames.append(pd.DataFrame({
            "evaluation": "loto",
            "heldout_type": heldout,
            "row_index": x_test.index,
            "y_true": y_test.to_numpy(),
            "score": loto_score,
            "threshold": threshold,
            "alarm": (loto_score >= threshold).astype(int),
        }))

        ref = all_test_pred.set_index("row_index").loc[x_test.index]
        ref_metric = safe_metrics(
            y_test,
            ref.score.to_numpy(),
            winner.threshold,
            y_test,
        )
        rows.append({
            "evaluation": "in_domain_reference",
            "source": "C",
            "heldout_type": heldout,
            "candidate": winner.candidate,
            "threshold": winner.threshold,
            **ref_metric,
        })
    return pd.DataFrame(rows), pd.concat(prediction_frames, ignore_index=True)


def figures(d_aggregate, c_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    x = np.arange(len(TYPES))
    width = 0.36
    for axis, metric, title, scale in (
        (axes[0], "macro_average_precision", "D available-mode macro AP", 1.0),
        (axes[1], "any_mode_normal_fpr", "D any-mode FPR (%)", 100.0),
    ):
        for offset, evaluation, color, label in (
            (-width / 2, "in_domain_reference", "#777777", "All-type reference"),
            (width / 2, "loto", "#5B0F78", "Leave-one-type-out"),
        ):
            frame = d_aggregate[
                (d_aggregate.feature_set == "engineered")
                & (d_aggregate.evaluation == evaluation)
            ].set_index("heldout_type").loc[list(TYPES)]
            axis.bar(x + offset, frame[metric] * scale, width, color=color, label=label)
        axis.set_xticks(x, TYPES)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 100)
    axes[0].legend(frameon=False)
    fig.suptitle("Large type shift exposes weaker cross-type generalization")
    fig.savefig(OUT / "loto_d_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "average_precision", "C binary AP"),
        (axes[1], "normal_fpr", "C normal FPR (%)"),
    ):
        for offset, evaluation, color, label in (
            (-width / 2, "in_domain_reference", "#777777", "All-type reference"),
            (width / 2, "loto", "#5B0F78", "Leave-one-type-out"),
        ):
            frame = c_metrics[
                c_metrics.evaluation == evaluation
            ].set_index("heldout_type").loc[list(TYPES)]
            values = frame[metric] * (100 if metric == "normal_fpr" else 1)
            axis.bar(x + offset, values, width, color=color, label=label)
        axis.set_xticks(x, TYPES)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 100)
    axes[0].legend(frameon=False)
    fig.suptitle("C supervised binary model under leave-one-type-out")
    fig.savefig(OUT / "loto_c_comparison.png", dpi=180)
    plt.close(fig)


def write_manifest(winner):
    manifest = {
        "heldout_types": list(TYPES),
        "protocol": (
            "Train on non-heldout Type rows; calibrate on non-heldout validation "
            "rows; test only on held-out Type rows."
        ),
        "d_target_fpr": TARGET_FPR,
        "c_candidate": winner.candidate,
        "note": (
            "Type one-hot categories unseen in a LOTO training fold encode as "
            "all-zero due to handle_unknown='ignore'."
        ),
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
    (OUT / "loto_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    config = read_config()
    data = load_data()
    d_metrics, d_aggregate, d_support, d_predictions = run_d(data, config)
    c_metrics, c_predictions = run_c(data)
    d_metrics.to_csv(
        OUT / "loto_d_mode_metrics.csv",
        index=False,
        float_format="%.17g",
    )
    d_aggregate.to_csv(
        OUT / "loto_d_aggregate.csv",
        index=False,
        float_format="%.17g",
    )
    d_support.to_csv(OUT / "loto_support.csv", index=False)
    d_predictions.to_csv(
        OUT / "loto_d_predictions.csv",
        index=False,
        float_format="%.17g",
    )
    c_metrics.to_csv(
        OUT / "loto_binary_metrics.csv",
        index=False,
        float_format="%.17g",
    )
    c_predictions.to_csv(
        OUT / "loto_binary_predictions.csv",
        index=False,
        float_format="%.17g",
    )
    figures(d_aggregate, c_metrics)
    winner, _ = selected_c_model()
    write_manifest(winner)
    print(
        "PASS: leave-one-product-type-out completed for D diagnostics and C binary model"
    )


if __name__ == "__main__":
    main()
