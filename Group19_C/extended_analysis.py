"""Prespecified evidence checks for C: paired uncertainty, alarm budgets, label use.

Run after C's freeze/test evaluation. The existing model choices are never changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score

from Group19_A.pipeline import fingerprint, load_split, split_fingerprint
from Group19_C.run import CONFIG, HERE, OUT, sha256

REPEATS = 2000
BOOTSTRAP_SEED = 20261123
BUDGETS = (0.01, 0.02, 0.05)
LABEL_FRACTIONS = (0.10, 0.25, 0.50, 1.00)
LABEL_SEEDS = (201, 202, 203, 204, 205)


def check_freeze():
    manifest = json.loads((OUT / "freeze_manifest.json").read_text(encoding="utf-8"))
    checks = (("data_sha256", fingerprint()), ("indices_sha256", split_fingerprint()),
              ("config_sha256", sha256(CONFIG)), ("code_sha256", sha256(HERE / "run.py")),
              ("models_sha256", sha256(OUT / "frozen_models.joblib")))
    for key, actual in checks:
        if manifest[key] != actual:
            raise ValueError(f"C freeze has changed: {key}")
    return manifest


def chosen_candidates():
    """Choose examples only by the already-saved validation AP."""
    c = pd.read_csv(OUT / "validation_metrics.csv")
    b_root = HERE.parent / "Group19_B" / "results"
    b_id = (b_root / "frozen" / "CURRENT").read_text(encoding="utf-8").strip()
    b = pd.read_csv(b_root / "frozen" / b_id / "metrics.csv")
    c_raw = c[c.feature_set == "raw"].sort_values("average_precision", ascending=False, kind="stable").iloc[0]
    c_engineered = c[c.feature_set == "engineered"].sort_values("average_precision", ascending=False, kind="stable").iloc[0]
    b_raw = b[b.feature_set == "raw"].sort_values("average_precision", ascending=False, kind="stable").iloc[0]
    raw_forest = c[(c.model == "RandomForest") & (c.feature_set == "raw") &
                   (c.weighting == "unweighted")].iloc[0]
    engineered_forest = c[(c.model == "RandomForest") & (c.feature_set == "engineered") &
                          (c.weighting == "unweighted")].iloc[0]
    assert json.loads(raw_forest.params) == json.loads(engineered_forest.params)
    return {"B_raw_validation_best": ("B", b_raw.candidate),
            "C_raw_validation_best": ("C", c_raw.candidate),
            "C_engineered_validation_best": ("C", c_engineered.candidate),
            "C_raw_forest": ("C", raw_forest.candidate),
            "C_engineered_forest": ("C", engineered_forest.candidate)}


def read_predictions(partition, choices):
    c = pd.read_csv(OUT / f"{partition}_predictions.csv", float_precision="round_trip")
    b_paths = list((HERE.parent / "Group19_B" / "results" / ("frozen" if partition == "validation" else "final_test")).glob(
        "*/predictions.csv"))
    if partition == "validation":
        b_id = (HERE.parent / "Group19_B" / "results" / "frozen" / "CURRENT").read_text(encoding="utf-8").strip()
        b_paths = [HERE.parent / "Group19_B" / "results" / "frozen" / b_id / "predictions.csv"]
    if len(b_paths) != 1:
        raise ValueError("Expected exactly one B prediction archive")
    b = pd.read_csv(b_paths[0], float_precision="round_trip")
    _, y, _ = load_split(partition, allow_test=partition == "test")
    expected = pd.Series(y.to_numpy(), index=y.index)
    result = {}
    for label, (source, key) in choices.items():
        table = c if source == "C" else b
        frame = table[table.candidate == key].set_index("row_index").sort_index()
        if len(frame) != len(expected) or not frame.index.is_unique or not frame.y_true.equals(expected.loc[frame.index].rename("y_true")):
            raise ValueError(f"Row or label mismatch for {label}")
        result[label] = frame
    return result


def paired_bootstrap(test):
    """Stratified paired row bootstrap; fixed models and observed class prevalence."""
    names = list(test)
    y = test[names[0]].y_true.to_numpy()
    scores = {name: test[name].score.to_numpy() for name in names}
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = {name: np.empty(REPEATS) for name in names}
    for repeat in range(REPEATS):
        sample = np.concatenate((rng.choice(positive, len(positive), replace=True),
                                 rng.choice(negative, len(negative), replace=True)))
        for name in names:
            draws[name][repeat] = average_precision_score(y[sample], scores[name][sample])
    rows = []
    for name in names:
        rows.append(dict(contrast=name, kind="model_ap", point=average_precision_score(y, scores[name]),
                         ci_low=np.quantile(draws[name], 0.025), ci_high=np.quantile(draws[name], 0.975),
                         bootstrap_repeats=REPEATS))
    for left, right in (("C_raw_validation_best", "B_raw_validation_best"),
                        ("C_engineered_forest", "C_raw_forest")):
        delta = draws[left] - draws[right]
        point = average_precision_score(y, scores[left]) - average_precision_score(y, scores[right])
        rows.append(dict(contrast=f"{left}_minus_{right}", kind="paired_ap_difference", point=point,
                         ci_low=np.quantile(delta, 0.025), ci_high=np.quantile(delta, 0.975),
                         bootstrap_repeats=REPEATS))
    pd.DataFrame(rows).to_csv(OUT / "paired_bootstrap.csv", index=False, float_format="%.17g")


def threshold_at_budget(y, scores, budget):
    """Lowest threshold allowing at most floor(budget * validation normals) FP."""
    normal = np.sort(np.asarray(scores)[np.asarray(y) == 0])[::-1]
    allowed = int(np.floor(budget * len(normal)))
    threshold = float(np.nextafter(normal[allowed], np.inf))
    if int(((np.asarray(scores) >= threshold) & (np.asarray(y) == 0)).sum()) > allowed:
        raise AssertionError("Validation budget exceeded")
    return threshold, allowed


def alarm_budgets(validation, test):
    rows = []
    for name in ("B_raw_validation_best", "C_raw_validation_best", "C_engineered_validation_best"):
        v, t = validation[name], test[name]
        for budget in BUDGETS:
            threshold, allowed = threshold_at_budget(v.y_true, v.score, budget)
            for partition, frame in (("validation", v), ("test", t)):
                alarm = frame.score.to_numpy() >= threshold
                y = frame.y_true.to_numpy()
                tp = int(((y == 1) & alarm).sum())
                fp = int(((y == 0) & alarm).sum())
                rows.append(dict(model=name, partition=partition, target_fpr=budget,
                                 allowed_validation_fp=allowed, threshold=threshold, tp=tp, fp=fp,
                                 alarms=tp + fp, recall=tp / int((y == 1).sum()),
                                 precision=tp / (tp + fp) if tp + fp else 0.0,
                                 actual_fpr=fp / int((y == 0).sum())))
    pd.DataFrame(rows).to_csv(OUT / "alarm_budgets.csv", index=False, float_format="%.17g")


def label_efficiency(choices):
    models = joblib.load(OUT / "frozen_models.joblib")
    rows = []
    for name in ("C_raw_validation_best", "C_engineered_validation_best"):
        key = choices[name][1]
        engineered = "engineered" in name
        x_train, y_train, _ = load_split("train", engineered=engineered)
        x_val, y_val, _ = load_split("validation", engineered=engineered)
        pos = np.flatnonzero(y_train.to_numpy() == 1)
        neg = np.flatnonzero(y_train.to_numpy() == 0)
        for fraction in LABEL_FRACTIONS:
            seeds = LABEL_SEEDS if fraction < 1 else LABEL_SEEDS[:1]
            for seed in seeds:
                if fraction == 1:
                    indices = np.arange(len(y_train))
                else:
                    rng = np.random.default_rng(seed)
                    indices = np.sort(np.concatenate((rng.permutation(pos)[:round(fraction * len(pos))],
                                                      rng.permutation(neg)[:round(fraction * len(neg))])))
                estimator = clone(models[key])
                estimator.fit(x_train.iloc[indices], y_train.iloc[indices])
                score = estimator.predict_proba(x_val)[:, 1]
                rows.append(dict(model=name, candidate=key, train_fraction=fraction, seed=seed,
                                 train_rows=len(indices), train_positives=int(y_train.iloc[indices].sum()),
                                 validation_ap=average_precision_score(y_val, score)))
                print(f"{name} fraction={fraction:g} seed={seed}: AP={rows[-1]['validation_ap']:.4f}", flush=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "label_efficiency_repeats.csv", index=False, float_format="%.17g")
    summary = detail.groupby(["model", "train_fraction"], as_index=False).agg(
        repeats=("validation_ap", "count"), train_rows=("train_rows", "first"),
        train_positives=("train_positives", "first"), ap_mean=("validation_ap", "mean"),
        ap_std=("validation_ap", "std"), ap_min=("validation_ap", "min"), ap_max=("validation_ap", "max"))
    summary["ap_std"] = summary.ap_std.fillna(0)
    summary.to_csv(OUT / "label_efficiency_summary.csv", index=False, float_format="%.17g")


def figures():
    budgets = pd.read_csv(OUT / "alarm_budgets.csv")
    test = budgets[budgets.partition == "test"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, frame in test.groupby("model", sort=False):
        ax.plot(frame.actual_fpr * 100, frame.recall * 100, marker="o", label=name.replace("_validation_best", ""))
        for _, row in frame.iterrows():
            ax.annotate(f"{row.target_fpr:.0%} target", (row.actual_fpr * 100, row.recall * 100),
                        xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set(xlabel="Realized test false positive rate (%)", ylabel="Test failure recall (%)",
           title="Alarm tradeoff at validation-calibrated budgets")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "alarm_budgets.png", dpi=180)
    plt.close(fig)
    summary = pd.read_csv(OUT / "label_efficiency_summary.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, frame in summary.groupby("model", sort=False):
        x, y, sd = frame.train_fraction.to_numpy() * 100, frame.ap_mean.to_numpy(), frame.ap_std.to_numpy()
        ax.errorbar(x, y, yerr=sd, marker="o", capsize=3, label=name.replace("_validation_best", ""))
    ax.set(xlabel="Fraction of labeled training rows used (%)", ylabel="Validation average precision",
           title="Label efficiency with frozen model settings", ylim=(0, 1))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "label_efficiency.png", dpi=180)
    plt.close(fig)


def main():
    check_freeze()
    choices = chosen_candidates()
    validation = read_predictions("validation", choices)
    test = read_predictions("test", choices)
    paired_bootstrap(test)
    alarm_budgets(validation, test)
    label_efficiency(choices)
    figures()
    print("PASS: paired bootstrap, alarm budgets, and label-efficiency analysis completed")


if __name__ == "__main__":
    main()
