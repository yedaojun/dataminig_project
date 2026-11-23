"""Rebuild C's figures from frozen result tables; no model tuning or test access."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

OUT = Path(__file__).resolve().parent / "results"


def main():
    metrics = pd.read_csv(OUT / "test_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, feature_set in zip(axes, ("raw", "engineered")):
        frame = metrics[metrics.feature_set == feature_set]
        x = np.arange(3)
        for offset, weighting, color in ((-0.18, "unweighted", "#2667a5"), (0.18, "weighted", "#e48a2a")):
            values = [float(frame[(frame.model == model) & (frame.weighting == weighting)].average_precision.iloc[0])
                      for model in ("LogisticRegression", "RandomForest", "XGBoost")]
            ax.bar(x + offset, values, width=0.34, label=weighting, color=color)
        ax.set_xticks(x, ("Logistic", "Forest", "XGBoost"))
        ax.set_title(f"{feature_set} inputs")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Test average precision")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=180)
    plt.close(fig)

    importance = pd.read_csv(OUT / "permutation_importance.csv")
    top = importance[(importance.model == "RandomForest") & (importance.feature_set == "engineered")]
    top = top.sort_values("ap_drop_mean").tail(9)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(top.feature, top.ap_drop_mean, xerr=top.ap_drop_std, color="#2667a5", alpha=0.85)
    ax.set_xlabel("Validation AP decrease when one feature is shuffled")
    ax.set_title("Engineered Random Forest: permutation importance")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=180)
    plt.close(fig)

    predictions = pd.read_csv(OUT / "test_predictions.csv", float_precision="round_trip")
    chosen = metrics[(metrics.weighting == "unweighted") & (metrics.model.isin(["RandomForest", "XGBoost"]))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for _, row in chosen.iterrows():
        p = predictions[predictions.candidate == row.candidate]
        precision, recall, _ = precision_recall_curve(p.y_true, p.score)
        fpr, tpr, _ = roc_curve(p.y_true, p.score)
        label = f"{row.model} / {row.feature_set}"
        axes[0].plot(recall, precision, label=label)
        axes[1].plot(fpr, tpr, label=label)
    axes[0].axhline(51 / 1500, color="gray", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Test precision–recall")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="Test ROC")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "pr_roc.png", dpi=180)
    plt.close(fig)
    print("PASS: three C figures rebuilt")


if __name__ == "__main__":
    main()
