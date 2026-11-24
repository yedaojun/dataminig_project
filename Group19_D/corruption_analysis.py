"""Gaussian, bias, and missing-sensor corruption comparison.

Run only in the isolated Conda environment because Group19_C freezes include
XGBoost:
    conda run -n aiaa3111_corruption python -B -m Group19_D.corruption_analysis

No model is refit or retuned. D diagnostics use their frozen per-mode
thresholds; C uses the validation-best frozen supervised pipeline; B uses its
validation-best raw LOF model.
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

from Group19_A.pipeline import MODES, NUM, features, load_split
from Group19_D.noise import add_gaussian_noise, verify_b_selection
from Group19_D.run import MODEL_FILE, OUT, mode_metrics, read_config, verify_freeze

REPO = Path(__file__).resolve().parent.parent
C_RESULTS = REPO / "Group19_C" / "results"
B_FROZEN = REPO / "Group19_B" / "results" / "frozen"
SEEDS = (101, 102, 103, 104, 105)
MISSING_FRACTION = 0.10

CONDITIONS = [
    ("clean", "clean", 0.0, (101,)),
    ("gaussian_0.1sd", "gaussian", 0.1, SEEDS),
    ("gaussian_0.2sd", "gaussian", 0.2, SEEDS),
    ("gaussian_0.5sd", "gaussian", 0.5, SEEDS),
    ("constant_bias_1sd", "constant_bias", 1.0, (101,)),
    ("missing_10pct_median", "random_missing", MISSING_FRACTION, SEEDS),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_reference():
    x_train, _, _ = load_split("train", engineered=False)
    scales = x_train[NUM].std(ddof=0)
    medians = x_train[NUM].median()
    return scales, medians


def corrupt(
    raw: pd.DataFrame,
    family: str,
    severity: float,
    seed: int,
    scales: pd.Series,
    medians: pd.Series,
) -> tuple[pd.DataFrame, int]:
    if family == "clean":
        return raw.copy(), 0
    if family == "gaussian":
        return add_gaussian_noise(raw, severity, seed, scales), 0
    if family == "constant_bias":
        biased = raw.copy()
        for column in NUM:
            biased[column] = biased[column] + severity * float(scales[column])
        return biased, 0
    if family == "random_missing":
        rng = np.random.default_rng(seed)
        missing = raw.copy()
        missing_count = 0
        for column in NUM:
            mask = rng.random(len(missing)) < severity
            missing_count += int(mask.sum())
            missing.loc[mask, column] = np.nan
            missing[column] = missing[column].fillna(float(medians[column]))
        return missing, missing_count
    raise ValueError(family)


def chosen_c_data():
    validation = pd.read_csv(
        C_RESULTS / "validation_metrics.csv",
        float_precision="round_trip",
    )
    winner = validation.sort_values(
        "average_precision",
        ascending=False,
        kind="stable",
    ).iloc[0]
    if winner.candidate != "engineered_RandomForest_unweighted_1":
        raise ValueError("Unexpected C validation winner")
    freeze = json.loads(
        (C_RESULTS / "freeze_manifest.json").read_text(encoding="utf-8")
    )
    selected = next(
        item for item in freeze["selected"] if item["candidate"] == winner.candidate
    )
    return winner, float(selected["threshold"])


def reference_scores(partition: str):
    references = {}
    d = pd.read_csv(
        OUT / f"{partition}_predictions.csv",
        float_precision="round_trip",
    )
    references["D"] = d

    c = pd.read_csv(
        C_RESULTS / f"{partition}_predictions.csv",
        float_precision="round_trip",
    )
    _, threshold = chosen_c_data()
    references["C"] = c[c.candidate == "engineered_RandomForest_unweighted_1"].copy()

    if partition == "test":
        b_paths = list((REPO / "Group19_B" / "results" / "final_test").glob("*/predictions.csv"))
        if len(b_paths) != 1:
            raise ValueError("Expected exactly one B final-test prediction file")
        references["B"] = pd.read_csv(
            b_paths[0],
            float_precision="round_trip",
        ).query("candidate == 'raw_LocalOutlierFactor_7'")
    else:
        digest = (B_FROZEN / "CURRENT").read_text(encoding="utf-8").strip()
        references["B"] = pd.read_csv(
            B_FROZEN / digest / "predictions.csv",
            float_precision="round_trip",
        ).query("candidate == 'raw_LocalOutlierFactor_7'")
    return references


def aligned_scores(reference: pd.DataFrame, index, column="score"):
    return reference.set_index("row_index").loc[index, column].to_numpy()


def run_experiment():
    config = read_config()
    verify_freeze()
    scales, medians = training_reference()
    d_models = joblib.load(MODEL_FILE)
    d_freeze = json.loads((OUT / "freeze_manifest.json").read_text(encoding="utf-8"))

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    c_models = joblib.load(C_RESULTS / "frozen_models.joblib")
    c_winner, c_threshold = chosen_c_data()
    c_pipeline = c_models[c_winner.candidate]

    b_folder, _, b_candidate, b_record = verify_b_selection(config)
    b_model = joblib.load(b_folder / "models.joblib")[b_candidate]

    x_test_raw, y_test, Y_test = load_split(
        "test",
        engineered=False,
        allow_test=True,
    )
    references = reference_scores("test")

    mode_rows = []
    binary_rows = []
    aggregate_rows = []
    for condition, family, severity, seeds in CONDITIONS:
        for seed in seeds:
            raw, missing_count = corrupt(
                x_test_raw,
                family,
                severity,
                seed,
                scales,
                medians,
            )
            engineered = features(raw, engineered=True)

            # Frozen D diagnostic models.
            for feature_set, model_input in (
                ("raw", raw),
                ("engineered", engineered),
            ):
                record = d_models[feature_set]
                matrix = record["preprocessor"].transform(model_input)
                thresholds = d_freeze["selected"][feature_set]["thresholds"]
                alarms = np.zeros((len(raw), len(MODES)), dtype=bool)
                mode_aps = []
                mode_f1 = []
                for mode_index, mode in enumerate(MODES):
                    score = record["models"][mode].predict_proba(matrix)[:, 1]
                    if condition == "clean":
                        reference = aligned_scores(
                            references["D"].query(
                                "feature_set == @feature_set and mode == @mode"
                            ),
                            raw.index,
                        )
                        if not np.allclose(score, reference, rtol=0, atol=1e-15):
                            raise AssertionError(
                                f"D clean replay differs: {feature_set}/{mode}"
                            )
                    alarm = score >= thresholds[mode]
                    alarms[:, mode_index] = alarm
                    metrics = mode_metrics(
                        Y_test[mode],
                        score,
                        thresholds[mode],
                        y_test,
                    )
                    mode_aps.append(metrics["average_precision"])
                    mode_f1.append(metrics["f1"])
                    mode_rows.append({
                        "condition": condition,
                        "condition_family": family,
                        "severity": severity,
                        "seed": seed,
                        "missing_cells": missing_count,
                        "source": "D",
                        "model": "OVR Random Forest",
                        "feature_set": feature_set,
                        "mode": mode,
                        **metrics,
                    })
                normal = y_test.to_numpy() == 0
                aggregate_rows.append({
                    "condition": condition,
                    "condition_family": family,
                    "severity": severity,
                    "seed": seed,
                    "feature_set": feature_set,
                    "macro_average_precision": float(np.mean(mode_aps)),
                    "macro_f1": float(np.mean(mode_f1)),
                    "any_mode_normal_fp": int(alarms[normal].any(axis=1).sum()),
                    "any_mode_normal_denominator": int(normal.sum()),
                    "any_mode_normal_fpr": float(alarms[normal].any(axis=1).mean()),
                })

            # Frozen C validation-best supervised model.
            c_score = c_pipeline.predict_proba(engineered)[:, 1]
            if condition == "clean":
                c_reference = aligned_scores(references["C"], raw.index)
                if not np.allclose(c_score, c_reference, rtol=0, atol=1e-12):
                    raise AssertionError("C clean replay differs")
            c_metrics = mode_metrics(y_test, c_score, c_threshold, y_test)
            binary_rows.append({
                "condition": condition,
                "condition_family": family,
                "severity": severity,
                "seed": seed,
                "missing_cells": missing_count,
                "source": "C",
                "model": c_winner.model,
                "candidate": c_winner.candidate,
                "feature_set": c_winner.feature_set,
                "mode": "TOTAL_FAILURE",
                **c_metrics,
            })

            # Frozen B validation-best raw LOF model.
            b_matrix = b_model["preprocessor"].transform(raw)
            if isinstance(b_matrix, pd.DataFrame):
                b_matrix = b_matrix.to_numpy()
            b_score = -b_model["model"].score_samples(b_matrix)
            if condition == "clean":
                b_reference = aligned_scores(references["B"], raw.index)
                if not np.allclose(b_score, b_reference, rtol=0, atol=1e-12):
                    raise AssertionError("B clean replay differs")
            b_metrics = mode_metrics(
                y_test,
                b_score,
                b_record["threshold"],
                y_test,
            )
            binary_rows.append({
                "condition": condition,
                "condition_family": family,
                "severity": severity,
                "seed": seed,
                "missing_cells": missing_count,
                "source": "B",
                "model": b_record["model"],
                "candidate": b_candidate,
                "feature_set": "raw",
                "mode": "TOTAL_FAILURE",
                **b_metrics,
            })
            print(
                f"{condition} seed={seed}: D raw/eng and C/B completed",
                flush=True,
            )

    mode_detail = pd.DataFrame(mode_rows)
    mode_detail.to_csv(
        OUT / "corruption_mode_repeats.csv",
        index=False,
        float_format="%.17g",
    )
    binary_detail = pd.DataFrame(binary_rows)
    binary_detail.to_csv(
        OUT / "corruption_binary_repeats.csv",
        index=False,
        float_format="%.17g",
    )
    aggregate_detail = pd.DataFrame(aggregate_rows)
    aggregate_detail.to_csv(
        OUT / "corruption_d_aggregate_repeats.csv",
        index=False,
        float_format="%.17g",
    )

    mode_summary = summarize(
        mode_detail,
        ["condition", "condition_family", "severity", "source", "model",
         "feature_set", "mode"],
    )
    mode_summary.to_csv(
        OUT / "corruption_mode_summary.csv",
        index=False,
        float_format="%.17g",
    )
    binary_summary = summarize(
        binary_detail,
        ["condition", "condition_family", "severity", "source", "model",
         "candidate", "feature_set"],
    )
    binary_summary.to_csv(
        OUT / "corruption_binary_summary.csv",
        index=False,
        float_format="%.17g",
    )
    aggregate_summary = summarize_aggregate(aggregate_detail)
    aggregate_summary.to_csv(
        OUT / "corruption_d_aggregate_summary.csv",
        index=False,
        float_format="%.17g",
    )
    return mode_detail, binary_detail, aggregate_detail, c_winner, b_record


def summarize(detail: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    summary = detail.groupby(keys, as_index=False).agg(
        repeats=("average_precision", "count"),
        support=("support", "first"),
        normal_denominator=("normal_denominator", "first"),
        average_precision_mean=("average_precision", "mean"),
        average_precision_std=("average_precision", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        normal_fpr_mean=("normal_fpr", "mean"),
        normal_fpr_std=("normal_fpr", "std"),
    )
    std_columns = [column for column in summary if column.endswith("_std")]
    summary[std_columns] = summary[std_columns].fillna(0.0)
    return summary


def summarize_aggregate(detail: pd.DataFrame) -> pd.DataFrame:
    summary = detail.groupby(
        ["condition", "condition_family", "severity", "feature_set"],
        as_index=False,
    ).agg(
        repeats=("macro_average_precision", "count"),
        macro_average_precision_mean=("macro_average_precision", "mean"),
        macro_average_precision_std=("macro_average_precision", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        any_mode_normal_fp_mean=("any_mode_normal_fp", "mean"),
        any_mode_normal_fp_std=("any_mode_normal_fp", "std"),
        any_mode_normal_fpr_mean=("any_mode_normal_fpr", "mean"),
        any_mode_normal_fpr_std=("any_mode_normal_fpr", "std"),
    )
    std_columns = [column for column in summary if column.endswith("_std")]
    summary[std_columns] = summary[std_columns].fillna(0.0)
    return summary


def figures(mode_summary, binary_summary, aggregate_summary):
    condition_order = [item[0] for item in CONDITIONS]
    labels = ["Clean", "N 0.1SD", "N 0.2SD", "N 0.5SD", "Bias +1SD", "Missing 10%"]
    x = np.arange(len(condition_order))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.7), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "average_precision_mean", "Binary average precision"),
        (axes[1], "f1_mean", "Binary F1 at frozen thresholds"),
    ):
        for source, color, marker in (
            ("C", "#5B0F78", "D"),
            ("B", "#777777", "o"),
        ):
            frame = binary_summary[binary_summary.source == source].set_index(
                "condition"
            ).loc[condition_order]
            axis.errorbar(
                x,
                frame[metric],
                yerr=frame[metric.replace("_mean", "_std")],
                marker=marker,
                linewidth=2,
                capsize=3,
                color=color,
                label="C engineered RF" if source == "C" else "B raw LOF",
            )
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        axis.set_ylim(bottom=0)
    axes[0].legend(frameon=False)
    fig.suptitle("Frozen binary models under distribution and missing-data corruption")
    fig.savefig(OUT / "corruption_binary_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.7), constrained_layout=True)
    for axis, metric, title, scale in (
        (axes[0], "macro_average_precision", "Five-mode macro AP", 1.0),
        (axes[1], "any_mode_normal_fpr", "Any-mode normal FPR (%)", 100.0),
    ):
        for feature_set, color, marker in (
            ("raw", "#777777", "o"),
            ("engineered", "#5B0F78", "D"),
        ):
            frame = aggregate_summary[
                aggregate_summary.feature_set == feature_set
            ].set_index("condition").loc[condition_order]
            axis.errorbar(
                x,
                frame[f"{metric}_mean"] * scale,
                yerr=frame[f"{metric}_std"] * scale,
                marker=marker,
                linewidth=2,
                capsize=3,
                color=color,
                label=feature_set.title(),
            )
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        axis.set_ylim(bottom=0)
    axes[0].legend(frameon=False)
    fig.suptitle("Frozen five-mode diagnostics under non-iid sensor corruption")
    fig.savefig(OUT / "corruption_d_comparison.png", dpi=180)
    plt.close(fig)

    engineered = mode_summary[
        (mode_summary.feature_set == "engineered")
        & (mode_summary["mode"].isin(MODES))
    ]
    matrix = np.empty((len(CONDITIONS), len(MODES)))
    for row, condition in enumerate(condition_order):
        frame = engineered[engineered.condition == condition].set_index("mode").loc[MODES]
        matrix[row] = frame.average_precision_mean.to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    image = ax.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap="Purples")
    ax.set_xticks(range(len(MODES)), MODES)
    ax.set_yticks(range(len(labels)), labels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
                fontsize=10,
            )
    ax.set_title("Engineered diagnostics: mode AP by corruption type")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    fig.tight_layout()
    fig.savefig(OUT / "corruption_mode_heatmap.png", dpi=180)
    plt.close(fig)


def write_manifest(c_winner, b_record):
    manifest = {
        "protocol": {
            "conditions": [
                {
                    "name": name,
                    "family": family,
                    "severity": severity,
                    "seeds": list(seeds),
                }
                for name, family, severity, seeds in CONDITIONS
            ],
            "missing_handling": (
                "Independent 10% cell missingness per numeric sensor; "
                "training medians used before recomputing engineered features."
            ),
            "no_refit_or_retune": True,
        },
        "models": {
            "D": "frozen raw and engineered OVR Random Forest",
            "C": {
                "candidate": c_winner.candidate,
                "model": c_winner.model,
                "feature_set": c_winner.feature_set,
            },
            "B": {
                "candidate": b_record["model"],
                "model": b_record["model"],
                "feature_set": "raw",
            },
        },
        "environment": {
            "python": platform.python_version(),
            **{
                package: importlib.metadata.version(package)
                for package in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "xgboost",
                    "joblib",
                )
            },
        },
        "source_sha256": {
            "C_models": sha256(C_RESULTS / "frozen_models.joblib"),
            "C_validation_metrics": sha256(C_RESULTS / "validation_metrics.csv"),
            "D_models": sha256(MODEL_FILE),
            "D_freeze_manifest": sha256(OUT / "freeze_manifest.json"),
        },
    }
    (OUT / "corruption_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    (
        mode_detail,
        binary_detail,
        aggregate_detail,
        c_winner,
        b_record,
    ) = run_experiment()
    mode_summary = pd.read_csv(OUT / "corruption_mode_summary.csv")
    binary_summary = pd.read_csv(OUT / "corruption_binary_summary.csv")
    aggregate_summary = pd.read_csv(OUT / "corruption_d_aggregate_summary.csv")
    figures(mode_summary, binary_summary, aggregate_summary)
    write_manifest(c_winner, b_record)
    print(
        "PASS: C/D/B frozen corruption comparison completed with "
        "Gaussian, +1-SD bias, and 10% random missingness"
    )


if __name__ == "__main__":
    main()
