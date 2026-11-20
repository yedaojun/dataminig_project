"""Render final curves from exported coordinates and operating points."""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(folder):
    curves = defaultdict(list)
    with (folder / "curves.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            curves[row["candidate"], row["curve"]].append((float(row["x"]), float(row["y"])))
    with (folder / "metrics.csv").open(newline="") as stream:
        metrics = list(csv.DictReader(stream))
    groups = [("Raw inputs | B/C primary", "raw"), ("Shared engineered inputs", "engineered"),
              ("Heat + LOF | B extension", "heat_fusion")]
    models = {"LocalOutlierFactor": ("LOF", "#c55318"), "IsolationForest": ("Isolation Forest", "#19669c"),
              "heat_lof_percentile_max": ("Fusion", "#247565")}
    for curve in ("PR", "ROC"):
        pr = curve == "PR"
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.65), sharey=True, layout="constrained")
        fig.suptitle(("Precision–recall" if pr else "ROC") + " on the held-out test set", fontsize=17)
        for ax, (title, feature_set) in zip(axes, groups):
            for row in metrics:
                if row["feature_set"] != feature_set:
                    continue
                count = {k: int(row[k]) for k in ("tp", "fp", "tn", "fn")}
                label, color = models[row["model"]]
                metric = float(row["average_precision" if pr else "roc_auc"])
                x, y = zip(*curves[row["candidate"], curve])
                ax.plot(x, y, color=color, drawstyle="steps-post" if pr else "default",
                        linewidth=1.5, label=f"{label}   {'AP' if pr else 'AUC'} {metric:.3f}")
                point = ((float(row["recall"]), float(row["precision"])) if pr else
                         (count["fp"] / (count["fp"] + count["tn"]), float(row["recall"])))
                ax.scatter(*point, color=color, s=36, edgecolors="white", linewidths=.7, zorder=4)
            if pr:
                prevalence = (count["tp"] + count["fn"]) / sum(count.values())
                ax.axhline(prevalence, color="gray", lw=1, ls="--", label=f"Prevalence   {prevalence:.3f}")
            else:
                ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--", label="Chance")
            ax.set(title=title, xlabel="Recall" if pr else "False positive rate",
                   xlim=(-.015, 1.015), ylim=(-.02, 1.05))
            ax.grid(alpha=.16)
            ax.legend(loc="upper right" if pr else "lower right", fontsize=8.5)
        axes[0].set_ylabel("Precision" if pr else "True positive rate")
        fig.supxlabel("Dots mark the thresholds selected on validation data.", fontsize=10)
        fig.savefig(folder / (curve.lower() + ".png"), dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    render(Path(sys.argv[1]))
