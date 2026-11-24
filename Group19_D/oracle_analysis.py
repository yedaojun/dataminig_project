"""Rule-based oracle and per-mode permutation importance.

Run after D freeze, test evaluation, and extended analysis:
    python -m Group19_D.oracle_analysis

The rule baseline is fixed from the AI4I generator description and is never
fitted. Permutation importance is computed on validation only, using the
already-frozen D models.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from Group19_A.pipeline import MODES, NUM, TARGET, load_split
from Group19_D.run import MODEL_FILE, OUT, read_config, verify_freeze

HERE = Path(__file__).resolve().parent
RULES_FILE = HERE / "oracle_rules.json"
IMPORTANCE_REPEATS = 5
IMPORTANCE_SEED = 20261124


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rule_scores(x: pd.DataFrame, rules: dict) -> dict[str, pd.Series | None]:
    temperature_difference = x["Process temperature [K]"] - x["Air temperature [K]"]
    power = x["Torque [Nm]"] * x["Rotational speed [rpm]"] * 2 * np.pi / 60
    overstrain = x["Tool wear [min]"] * x["Torque [Nm]"]
    thresholds = rules["modes"]["OSF"]["thresholds"]
    return {
        "TWF": x["Tool wear [min]"].between(200, 240, inclusive="both").astype(int),
        "HDF": (
            (temperature_difference < 8.6)
            & (x["Rotational speed [rpm]"] < 1380)
        ).astype(int),
        "PWF": ((power < 3500) | (power > 9000)).astype(int),
        "OSF": (overstrain > x.Type.map(thresholds)).astype(int),
        "RNF": None,
    }


def metrics(y, prediction, total_failure) -> dict:
    y = np.asarray(y, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    total_failure = np.asarray(total_failure, dtype=int)
    tp = int(np.sum((y == 1) & (prediction == 1)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    tn = int(np.sum((y == 0) & (prediction == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    normal = total_failure == 0
    normal_fp = int(np.sum(normal & (prediction == 1)))
    return {
        "average_precision": (
            float(average_precision_score(y, prediction))
            if 0 < y.sum() < len(y)
            else np.nan
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "support": int(y.sum()),
        "normal_fp": normal_fp,
        "normal_denominator": int(normal.sum()),
        "normal_fpr": normal_fp / int(normal.sum()),
        "exact_match_rate": float(np.mean(y == prediction)),
    }


def oracle_evaluation(rules: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    prediction_frames = []
    for partition in ("train", "validation", "test"):
        x, total_failure, labels = load_split(
            partition,
            allow_test=partition == "test",
        )
        scores = rule_scores(x, rules)
        for mode in MODES:
            prediction = scores[mode]
            available = prediction is not None
            candidate = (
                prediction.to_numpy(dtype=int)
                if available
                else np.zeros(len(x), dtype=int)
            )
            values = metrics(labels[mode], candidate, total_failure)
            if not available:
                values["average_precision"] = np.nan
            spec = rules["modes"][mode]
            rows.append({
                "partition": partition,
                "source": "rule_oracle",
                "mode": mode,
                "available": available,
                "deterministic_label": spec["deterministic_label"],
                "rule": spec["rule"],
                **values,
            })
            prediction_frames.append(pd.DataFrame({
                "partition": partition,
                "row_index": x.index,
                "mode": mode,
                "rule_available": available,
                "rule_score": candidate,
                "y_true": labels[mode].to_numpy(),
                "total_failure": total_failure.to_numpy(),
            }))

        binary_rule = (
            scores["TWF"]
            | scores["HDF"]
            | scores["PWF"]
            | scores["OSF"]
        ).astype(int)
        binary_metrics = metrics(total_failure, binary_rule, total_failure)
        rows.append({
            "partition": partition,
            "source": "rule_oracle",
            "mode": "TOTAL_FAILURE",
            "available": True,
            "deterministic_label": False,
            "rule": "TWF-or-HDF-or-PWF-or-OSF documented rules; RNF unavailable",
            **binary_metrics,
        })
        prediction_frames.append(pd.DataFrame({
            "partition": partition,
            "row_index": x.index,
            "mode": "TOTAL_FAILURE",
            "rule_available": True,
            "rule_score": binary_rule.to_numpy(),
            "y_true": total_failure.to_numpy(),
            "total_failure": total_failure.to_numpy(),
        }))
    detail = pd.DataFrame(rows)
    detail.to_csv(
        OUT / "oracle_metrics.csv",
        index=False,
        float_format="%.17g",
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(
        OUT / "oracle_predictions.csv",
        index=False,
        float_format="%.17g",
    )
    return detail, predictions


def oracle_comparison(oracle: pd.DataFrame) -> pd.DataFrame:
    d = pd.read_csv(
        OUT / "test_metrics.csv",
        float_precision="round_trip",
    )
    test_oracle = oracle[oracle.partition == "test"].copy()
    comparison = d.merge(
        test_oracle[
            [
                "mode",
                "available",
                "deterministic_label",
                "average_precision",
                "precision",
                "recall",
                "f1",
                "exact_match_rate",
            ]
        ].rename(columns={
            "average_precision": "oracle_average_precision",
            "precision": "oracle_precision",
            "recall": "oracle_recall",
            "f1": "oracle_f1",
            "exact_match_rate": "oracle_exact_match_rate",
        }),
        on="mode",
        how="left",
        suffixes=("", "_d"),
    )
    comparison["oracle_interpretation"] = np.select(
        [
            comparison.deterministic_label == True,
            comparison["mode"] == "TWF",
            comparison["mode"] == "RNF",
        ],
        [
            "exact_generator_rule",
            "stochastic_exposure_window",
            "no_deterministic_rule",
        ],
        default="unknown",
    )
    comparison.to_csv(
        OUT / "oracle_vs_d.csv",
        index=False,
        float_format="%.17g",
    )
    return comparison


def permutation_importance(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    verify_freeze()
    frozen = joblib.load(MODEL_FILE)
    rows = []
    feature_set_seed = {"raw": 0, "engineered": 1000}
    for feature_set in config["feature_sets"]:
        engineered = feature_set == "engineered"
        x_val, _, labels = load_split("validation", engineered=engineered)
        record = frozen[feature_set]
        matrix = record["preprocessor"].transform(x_val)
        for mode_index, mode in enumerate(MODES):
            y = labels[mode].to_numpy()
            baseline = record["models"][mode].predict_proba(matrix)[:, 1]
            baseline_ap = float(average_precision_score(y, baseline))
            for column_index, column in enumerate(x_val.columns):
                values = []
                for repeat in range(IMPORTANCE_REPEATS):
                    rng = np.random.default_rng(
                        IMPORTANCE_SEED
                        + feature_set_seed[feature_set]
                        + mode_index * 100
                        + column_index * 10
                        + repeat
                    )
                    permuted = x_val.copy()
                    permuted[column] = rng.permutation(
                        permuted[column].to_numpy()
                    )
                    transformed = record["preprocessor"].transform(permuted)
                    score = record["models"][mode].predict_proba(transformed)[:, 1]
                    permuted_ap = float(average_precision_score(y, score))
                    values.append(baseline_ap - permuted_ap)
                rows.append({
                    "feature_set": feature_set,
                    "mode": mode,
                    "feature": column,
                    "validation_support": int(y.sum()),
                    "baseline_ap": baseline_ap,
                    "repeats": IMPORTANCE_REPEATS,
                    "ap_drop_mean": float(np.mean(values)),
                    "ap_drop_std": float(np.std(values, ddof=1)),
                    "ap_drop_min": float(np.min(values)),
                    "ap_drop_max": float(np.max(values)),
                    "interpretation": (
                        "insufficient_support" if mode == "RNF" else "descriptive"
                    ),
                })
            print(
                f"permutation {feature_set}/{mode}: baseline AP={baseline_ap:.5f}",
                flush=True,
            )
    detail = pd.DataFrame(rows)
    detail.to_csv(
        OUT / "permutation_importance_by_mode.csv",
        index=False,
        float_format="%.17g",
    )
    detail["rank_within_mode"] = detail.groupby(
        ["feature_set", "mode"]
    ).ap_drop_mean.rank(ascending=False, method="first").astype(int)
    summary = detail.sort_values(
        ["feature_set", "mode", "rank_within_mode"]
    ).copy()
    summary.to_csv(
        OUT / "permutation_importance_summary.csv",
        index=False,
        float_format="%.17g",
    )
    return detail, summary


def figures(comparison: pd.DataFrame, importance: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    positions = np.arange(len(MODES))
    width = 0.26
    oracle = comparison.groupby("mode", sort=False).first().loc[MODES]
    raw = comparison[comparison.feature_set == "raw"].set_index("mode").loc[MODES]
    engineered = (
        comparison[comparison.feature_set == "engineered"]
        .set_index("mode")
        .loc[MODES]
    )
    ax.bar(
        positions - width,
        oracle.oracle_average_precision.astype(float),
        width,
        color="#C8A34A",
        label="Rule oracle",
    )
    ax.bar(
        positions,
        raw.average_precision,
        width,
        color="#777777",
        label="D raw RF",
    )
    ax.bar(
        positions + width,
        engineered.average_precision,
        width,
        color="#5B0F78",
        label="D engineered RF",
    )
    ax.bar(
        positions[4],
        engineered.average_precision.iloc[4],
        width,
        color="#5B0F78",
        edgecolor="black",
        hatch="///",
    )
    ax.text(
        positions[4] - width,
        0.03,
        "no rule",
        ha="center",
        va="bottom",
        rotation=90,
        fontsize=9,
    )
    ax.set_xticks(positions, MODES)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Test average precision")
    ax.set_title("Generator-rule oracle vs frozen diagnostic models")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "oracle_vs_d.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 4.8),
        constrained_layout=True,
    )
    for axis, feature_set in zip(axes, ("raw", "engineered")):
        frame = importance[importance.feature_set == feature_set]
        features = list(dict.fromkeys(frame.feature))
        matrix = np.full((len(MODES), len(features)), np.nan)
        for mode_index, mode in enumerate(MODES):
            values = (
                frame[frame["mode"] == mode]
                .set_index("feature")
                .reindex(features)
            )
            matrix[mode_index] = values.ap_drop_mean.to_numpy()
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap="Purples",
            vmin=0,
            vmax=max(0.001, np.nanmax(matrix)),
        )
        axis.set_xticks(
            range(len(features)),
            [feature.replace(" [", "\n[").replace("temperature_difference", "temp_diff")
             .replace("mechanical_power", "power").replace("wear_load", "wear_load")
             for feature in features],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        axis.set_yticks(range(len(MODES)), MODES)
        axis.set_title(f"{feature_set}: validation AP drop")
        fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    fig.suptitle("Per-mode permutation importance (5 repeats; RNF n=4 in validation)")
    fig.savefig(OUT / "mode_feature_importance_heatmap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    config = read_config()
    rules = json.loads(RULES_FILE.read_text(encoding="utf-8"))
    oracle, _ = oracle_evaluation(rules)
    comparison = oracle_comparison(oracle)
    importance_detail, importance_summary = permutation_importance(config)
    figures(comparison, importance_detail)
    manifest = {
        "source": "AI4I generator rules; no rule fitted on data",
        "rules_file": RULES_FILE.name,
        "rules_sha256": sha256(RULES_FILE),
        "importance_partition": "validation",
        "importance_repeats": IMPORTANCE_REPEATS,
        "importance_seed": IMPORTANCE_SEED,
        "deterministic_rules": ["HDF", "PWF", "OSF"],
        "stochastic_or_unavailable": ["TWF", "RNF"],
        "outputs": {
            "oracle_metrics": "oracle_metrics.csv",
            "oracle_predictions": "oracle_predictions.csv",
            "oracle_vs_d": "oracle_vs_d.csv",
            "importance": "permutation_importance_by_mode.csv",
            "importance_summary": "permutation_importance_summary.csv",
            "oracle_figure": "oracle_vs_d.png",
            "importance_figure": "mode_feature_importance_heatmap.png",
        },
    }
    (OUT / "oracle_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "PASS: generator-rule oracle and five-mode permutation importance completed"
    )


if __name__ == "__main__":
    main()
