"""Sensor-noise robustness for frozen D models and one preselected B baseline.

Run after `python -m Group19_D.run evaluate-test`:
    python -m Group19_D.noise
"""
from __future__ import annotations

import json
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
from Group19_D.run import (
    MODEL_FILE,
    OUT,
    mode_metrics,
    read_config,
    verify_freeze,
)

REPO = Path(__file__).resolve().parent.parent
B_FROZEN = REPO / "Group19_B" / "results" / "frozen"


def load_training_scales() -> pd.Series:
    x_train, _, _ = load_split("train", engineered=False)
    scales = x_train[NUM].std(ddof=0)
    recorded = pd.read_csv(
        REPO / "Group19_A" / "results" / "training_sensor_std.csv",
        index_col=0,
    )["training_std_ddof0"]
    recorded = recorded.reindex(NUM)
    if not np.allclose(scales.to_numpy(), recorded.to_numpy(), rtol=0, atol=1e-12):
        raise ValueError("Training sensor scales differ from A's recorded values")
    return scales


def add_gaussian_noise(raw_features: pd.DataFrame, multiplier: float,
                       seed: int, scales: pd.Series) -> pd.DataFrame:
    if multiplier == 0:
        return raw_features.copy()
    rng = np.random.default_rng(seed)
    noisy = raw_features.copy()
    for column in NUM:
        noisy[column] = noisy[column] + rng.normal(
            loc=0.0,
            scale=multiplier * float(scales[column]),
            size=len(noisy),
        )
    return noisy


def input_for_feature_set(noisy_raw: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set == "raw":
        return noisy_raw
    if feature_set == "engineered":
        return features(noisy_raw, engineered=True)
    raise ValueError(feature_set)


def summarize(detail: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    summary = detail.groupby(keys, as_index=False).agg(
        repeats=("average_precision", "count"),
        support=("support", "first"),
        normal_denominator=("normal_denominator", "first"),
        average_precision_mean=("average_precision", "mean"),
        average_precision_std=("average_precision", "std"),
        average_precision_min=("average_precision", "min"),
        average_precision_max=("average_precision", "max"),
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


def d_noise_experiment(
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = verify_freeze()
    frozen = joblib.load(MODEL_FILE)
    scales = load_training_scales()
    pd.DataFrame({
        "sensor": NUM,
        "training_std_ddof0": scales.to_numpy(),
    }).to_csv(OUT / "noise_standard_deviations.csv", index=False, float_format="%.17g")

    x_test_raw, y_test, Y_test = load_split(
        "test",
        engineered=False,
        allow_test=True,
    )
    baseline = pd.read_csv(
        OUT / "test_predictions.csv",
        float_precision="round_trip",
    )
    rows = []
    aggregate_rows = []
    normal = y_test.to_numpy() == 0
    for feature_set in config["feature_sets"]:
        record = frozen[feature_set]
        thresholds = manifest["selected"][feature_set]["thresholds"]
        for multiplier in config["noise"]["multipliers"]:
            for seed in config["noise"]["seeds"]:
                noisy_raw = add_gaussian_noise(
                    x_test_raw,
                    float(multiplier),
                    int(seed),
                    scales,
                )
                model_input = input_for_feature_set(noisy_raw, feature_set)
                matrix = record["preprocessor"].transform(model_input)
                modes_for_frame = []
                any_alarm = np.zeros(len(x_test_raw), dtype=bool)
                for mode in MODES:
                    scores = record["models"][mode].predict_proba(matrix)[:, 1]
                    if multiplier == 0:
                        reference = (
                            baseline[
                                (baseline.feature_set == feature_set)
                                & (baseline["mode"] == mode)
                            ]
                            .sort_values("row_index")
                            .score
                            .to_numpy()
                        )
                        row_order = np.argsort(
                            x_test_raw.index.to_numpy(),
                            kind="mergesort",
                        )
                        if not np.allclose(
                            scores[row_order],
                            reference,
                            rtol=0,
                            atol=1e-15,
                        ):
                            raise AssertionError(
                                f"Zero-noise replay differs from frozen test rows: "
                                f"{feature_set}/{mode}/seed={seed}"
                            )
                    metrics = mode_metrics(
                        Y_test[mode],
                        scores,
                        thresholds[mode],
                        y_test,
                    )
                    any_alarm |= scores >= thresholds[mode]
                    rows.append({
                        "feature_set": feature_set,
                        "multiplier": float(multiplier),
                        "seed": int(seed),
                        "mode": mode,
                        **metrics,
                    })
                    modes_for_frame.append(metrics)
                aggregate_rows.append({
                    "feature_set": feature_set,
                    "multiplier": float(multiplier),
                    "seed": int(seed),
                    "macro_average_precision": float(np.mean(
                        [metric["average_precision"] for metric in modes_for_frame]
                    )),
                    "macro_precision": float(np.mean(
                        [metric["precision"] for metric in modes_for_frame]
                    )),
                    "macro_recall": float(np.mean(
                        [metric["recall"] for metric in modes_for_frame]
                    )),
                    "macro_f1": float(np.mean(
                        [metric["f1"] for metric in modes_for_frame]
                    )),
                    "any_mode_normal_fp": int(any_alarm[normal].sum()),
                    "any_mode_normal_denominator": int(normal.sum()),
                    "any_mode_normal_fpr": float(any_alarm[normal].mean()),
                })
                print(
                    f"{feature_set} multiplier={multiplier:g} seed={seed}: "
                    f"macro AP={np.mean([m['average_precision'] for m in modes_for_frame]):.6f}",
                    flush=True,
                )

    detail = pd.DataFrame(rows)
    detail.to_csv(
        OUT / "noise_repeats.csv",
        index=False,
        float_format="%.17g",
    )
    summary = summarize(detail, ["feature_set", "multiplier", "mode"])
    summary.to_csv(
        OUT / "noise_summary.csv",
        index=False,
        float_format="%.17g",
    )
    aggregate_detail = pd.DataFrame(aggregate_rows)
    aggregate_detail.to_csv(
        OUT / "noise_aggregate_repeats.csv",
        index=False,
        float_format="%.17g",
    )
    aggregate_summary = aggregate_detail.groupby(
        ["feature_set", "multiplier"],
        as_index=False,
    ).agg(
        repeats=("macro_average_precision", "count"),
        macro_average_precision_mean=("macro_average_precision", "mean"),
        macro_average_precision_std=("macro_average_precision", "std"),
        macro_precision_mean=("macro_precision", "mean"),
        macro_precision_std=("macro_precision", "std"),
        macro_recall_mean=("macro_recall", "mean"),
        macro_recall_std=("macro_recall", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        any_mode_normal_fp_mean=("any_mode_normal_fp", "mean"),
        any_mode_normal_fp_std=("any_mode_normal_fp", "std"),
        any_mode_normal_fpr_mean=("any_mode_normal_fpr", "mean"),
        any_mode_normal_fpr_std=("any_mode_normal_fpr", "std"),
    )
    aggregate_summary = aggregate_summary.fillna(0.0)
    aggregate_summary.to_csv(
        OUT / "noise_aggregate_summary.csv",
        index=False,
        float_format="%.17g",
    )
    return detail, summary, aggregate_detail, aggregate_summary


def verify_b_selection(config: dict) -> tuple[Path, dict, str, dict]:
    if not B_FROZEN.is_dir():
        raise FileNotFoundError("Group19_B frozen artifact directory is missing")
    digest = (B_FROZEN / "CURRENT").read_text(encoding="utf-8").strip()
    folder = B_FROZEN / digest
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    raw = {
        candidate: record
        for candidate, record in manifest["selected"].items()
        if record["feature_set"] == "raw"
    }
    validation_best = max(
        raw,
        key=lambda candidate: raw[candidate]["validation_metrics"]["average_precision"],
    )
    expected = config["noise"]["binary_supplement"]["candidate"]
    if expected != validation_best:
        raise ValueError(
            f"Configured B supplement {expected} is not the validation-selected "
            f"raw model {validation_best}"
        )
    return folder, manifest, validation_best, raw[validation_best]


def b_noise_experiment(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder, manifest, candidate, record = verify_b_selection(config)
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    fitted = joblib.load(folder / "models.joblib")[candidate]
    x_test_raw, y_test, _ = load_split(
        "test",
        engineered=False,
        allow_test=True,
    )
    scales = load_training_scales()
    rows = []
    for multiplier in config["noise"]["multipliers"]:
        for seed in config["noise"]["seeds"]:
            noisy_raw = add_gaussian_noise(
                x_test_raw,
                float(multiplier),
                int(seed),
                scales,
            )
            matrix = fitted["preprocessor"].transform(noisy_raw)
            if isinstance(matrix, pd.DataFrame):
                matrix = matrix.to_numpy()
            scores = -fitted["model"].score_samples(matrix)
            metrics = mode_metrics(
                y_test,
                scores,
                record["threshold"],
                y_test,
            )
            rows.append({
                "source": "B",
                "model": record["model"],
                "candidate": candidate,
                "feature_set": "raw",
                "multiplier": float(multiplier),
                "seed": int(seed),
                **metrics,
            })
            print(
                f"B binary multiplier={multiplier:g} seed={seed}: "
                f"AP={metrics['average_precision']:.6f}",
                flush=True,
            )
    detail = pd.DataFrame(rows)
    detail.to_csv(
        OUT / "binary_noise_repeats.csv",
        index=False,
        float_format="%.17g",
    )
    summary = summarize(
        detail.rename(
            columns={
                "average_precision": "average_precision",
            }
        ),
        ["source", "model", "candidate", "feature_set", "multiplier"],
    )
    summary.to_csv(
        OUT / "binary_noise_summary.csv",
        index=False,
        float_format="%.17g",
    )
    return detail, summary


def plot_d_noise(summary: pd.DataFrame) -> None:
    colors = {
        "TWF": "#1f77b4",
        "HDF": "#ff7f0e",
        "PWF": "#2ca02c",
        "OSF": "#d62728",
        "RNF": "#9467bd",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, feature_set in enumerate(("raw", "engineered")):
        frame = summary[summary.feature_set == feature_set]
        for mode in MODES:
            group = frame[frame["mode"] == mode].sort_values("multiplier")
            x = group.multiplier.to_numpy() * 100
            axes[row, 0].errorbar(
                x,
                group.average_precision_mean,
                yerr=group.average_precision_std,
                marker="o",
                capsize=3,
                label=mode,
                color=colors[mode],
            )
            axes[row, 1].errorbar(
                x,
                group.f1_mean,
                yerr=group.f1_std,
                marker="o",
                capsize=3,
                label=mode,
                color=colors[mode],
            )
        axes[row, 0].set_title(f"{feature_set}: average precision")
        axes[row, 1].set_title(f"{feature_set}: F1 at frozen thresholds")
        for axis in axes[row]:
            axis.set_xlabel("Gaussian sensor-noise multiplier (% of training SD)")
            axis.set_ylim(0, 1.03)
            axis.grid(alpha=0.2)
    axes[0, 0].legend(title="Mode", fontsize=8)
    fig.suptitle("Group 19 D: test robustness under recalibrated engineered features")
    fig.savefig(OUT / "noise_robustness.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, feature_set in zip(axes, ("raw", "engineered")):
        frame = summary[summary.feature_set == feature_set]
        for mode in MODES:
            group = frame[frame["mode"] == mode].sort_values("multiplier")
            axis.errorbar(
                group.multiplier.to_numpy() * 100,
                group.normal_fpr_mean,
                yerr=group.normal_fpr_std,
                marker="o",
                capsize=3,
                label=mode,
                color=colors[mode],
            )
        axis.set(
            title=f"{feature_set}: realized FPR on test total-failure normals",
            xlabel="Gaussian sensor-noise multiplier (% of training SD)",
            ylabel="False positive rate",
        )
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
    axes[0].legend(title="Mode", fontsize=8)
    fig.savefig(OUT / "noise_fpr.png", dpi=180)
    plt.close(fig)


def plot_b_noise(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, metric, label in (
        (axes[0], "average_precision_mean", "Average precision"),
        (axes[1], "f1_mean", "F1 at frozen threshold"),
    ):
        axis.errorbar(
            summary.multiplier.to_numpy() * 100,
            summary[metric],
            yerr=summary[metric.replace("_mean", "_std")],
            marker="o",
            capsize=3,
            color="#7f2704",
        )
        axis.set(
            xlabel="Gaussian sensor-noise multiplier (% of training SD)",
            ylabel=label,
            title=f"B raw LOF binary supplement: {label}",
        )
        axis.set_ylim(0, 1.03)
        axis.grid(alpha=0.2)
    fig.savefig(OUT / "binary_noise_robustness.png", dpi=180)
    plt.close(fig)


def main() -> None:
    config = read_config()
    _, d_summary, _, _ = d_noise_experiment(config)
    _, b_summary = b_noise_experiment(config)
    plot_d_noise(d_summary)
    plot_b_noise(b_summary)
    print(
        "PASS: D per-mode noise repeats, summaries, and B binary supplement "
        "completed without refitting or retuning"
    )


if __name__ == "__main__":
    main()
