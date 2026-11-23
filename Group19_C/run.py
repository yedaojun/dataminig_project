"""Supervised binary detection using A's frozen split and B's comparison contract.

Run from the repository root: python -m Group19_C.run tune / evaluate-test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline

from Group19_A.pipeline import fingerprint, load_split, make_preprocessor, split_fingerprint

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
CONFIG = HERE / "config.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def threshold_max_f1(y, score):
    """Observed validation score with maximal F1; ties use the higher score."""
    order = np.argsort(-score, kind="mergesort")
    s, labels = np.asarray(score)[order], np.asarray(y)[order]
    tp = fp = 0
    positives = int(labels.sum())
    best_f1, best_threshold = -1.0, None
    for i, (value, label) in enumerate(zip(s, labels)):
        tp += int(label == 1)
        fp += int(label == 0)
        if i + 1 < len(s) and s[i + 1] == value:
            continue
        f1 = 2 * tp / (positives + tp + fp) if positives + tp + fp else 0.0
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(value)
    return best_threshold


def metrics(y, score, threshold):
    alarm = np.asarray(score) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, alarm, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(y, alarm, average="binary", zero_division=0)
    return dict(average_precision=float(average_precision_score(y, score)),
                roc_auc=float(roc_auc_score(y, score)), threshold=float(threshold),
                precision=float(precision), recall=float(recall), f1=float(f1),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def make_model(name, params, weighted, ratio, seed):
    if name == "LogisticRegression":
        return LogisticRegression(C=params["C"], class_weight="balanced" if weighted else None,
                                  max_iter=3000, random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=150, max_depth=params["max_depth"],
                                      min_samples_leaf=2, class_weight="balanced" if weighted else None,
                                      n_jobs=-1, random_state=seed)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200, max_depth=params["max_depth"], learning_rate=0.05,
                             subsample=1.0, colsample_bytree=1.0, objective="binary:logistic",
                             scale_pos_weight=ratio if weighted else 1.0,
                             tree_method="hist", n_jobs=4, random_state=seed, eval_metric="logloss")
    raise ValueError(name)


def prediction_rows(partition, key, model, feature_set, y, score, threshold):
    return pd.DataFrame(dict(partition=partition, row_index=y.index.to_numpy(), candidate=key,
                             model=model, feature_set=feature_set, y_true=y.to_numpy(),
                             score=score, threshold=threshold, alarm=(score >= threshold).astype(int)))


def tune():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg["seed"] != 42 or cfg["primary_metric"] != "average_precision":
        raise ValueError("C configuration no longer matches the shared A/B protocol")
    OUT.mkdir(exist_ok=True)
    ratio = None
    candidate_rows, selected_rows, validation_rows, fitted = [], [], [], {}
    for feature_set in cfg["feature_sets"]:
        engineered = feature_set == "engineered"
        x_train, y_train, _ = load_split("train", engineered=engineered)
        x_val, y_val, _ = load_split("validation", engineered=engineered)
        ratio = float((y_train == 0).sum() / (y_train == 1).sum())
        for name, grid in cfg["candidates"].items():
            for weighting in cfg["weighting"]:
                best = None
                for ordinal, params in enumerate(grid):
                    key = f"{feature_set}_{name}_{weighting}_{ordinal}"
                    pipe = Pipeline([("preprocess", make_preprocessor(engineered, scale=name == "LogisticRegression")),
                                     ("model", make_model(name, params, weighting == "weighted", ratio, cfg["seed"]))])
                    pipe.fit(x_train, y_train)
                    score = pipe.predict_proba(x_val)[:, 1]
                    ap = float(average_precision_score(y_val, score))
                    auc = float(roc_auc_score(y_val, score))
                    candidate_rows.append(dict(candidate=key, model=name, feature_set=feature_set,
                                               weighting=weighting, params=json.dumps(params, sort_keys=True),
                                               validation_ap=ap, validation_roc_auc=auc))
                    if best is None or ap > best["ap"]:
                        best = dict(key=key, pipe=pipe, score=score, ap=ap, params=params)
                    print(f"{key}: validation AP={ap:.5f}", flush=True)
                threshold = threshold_max_f1(y_val, best["score"])
                summary = dict(candidate=best["key"], model=name, feature_set=feature_set,
                               weighting=weighting, params=json.dumps(best["params"], sort_keys=True),
                               partition="validation", **metrics(y_val, best["score"], threshold))
                selected_rows.append(summary)
                fitted[best["key"]] = best["pipe"]
                validation_rows.append(prediction_rows("validation", best["key"], name, feature_set,
                                                       y_val, best["score"], threshold))
    pd.DataFrame(candidate_rows).to_csv(OUT / "candidates.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(OUT / "validation_metrics.csv", index=False)
    pd.concat(validation_rows, ignore_index=True).to_csv(OUT / "validation_predictions.csv", index=False, float_format="%.17g")
    joblib.dump(fitted, OUT / "frozen_models.joblib")
    manifest = dict(status="frozen_before_test", data_sha256=fingerprint(), indices_sha256=split_fingerprint(),
                    config_sha256=sha256(CONFIG), code_sha256=sha256(Path(__file__)),
                    models_sha256=sha256(OUT / "frozen_models.joblib"), seed=cfg["seed"],
                    class_ratio=ratio, sklearn_version=sklearn.__version__,
                    threshold_rule=cfg["threshold_rule"], score_direction="higher_means_failure",
                    selected=[dict(candidate=r["candidate"], model=r["model"], feature_set=r["feature_set"],
                                   weighting=r["weighting"], params=r["params"], threshold=r["threshold"])
                              for r in selected_rows])
    (OUT / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("PASS: frozen 12 validation-selected models and thresholds; test not accessed")


def evaluate_test():
    manifest = json.loads((OUT / "freeze_manifest.json").read_text(encoding="utf-8"))
    checks = (("data_sha256", fingerprint()), ("indices_sha256", split_fingerprint()),
              ("config_sha256", sha256(CONFIG)), ("code_sha256", sha256(Path(__file__))),
              ("models_sha256", sha256(OUT / "frozen_models.joblib")))
    for field, actual in checks:
        if manifest[field] != actual:
            raise ValueError(f"Frozen {field} changed; re-tune and re-freeze before test")
    fitted = joblib.load(OUT / "frozen_models.joblib")
    all_rows, summaries = [], []
    for item in manifest["selected"]:
        key = item["candidate"]
        engineered = item["feature_set"] == "engineered"
        x_test, y_test, _ = load_split("test", engineered=engineered, allow_test=True)
        score = fitted[key].predict_proba(x_test)[:, 1]
        threshold = float(item["threshold"])
        all_rows.append(prediction_rows("test", key, item["model"], item["feature_set"],
                                        y_test, score, threshold))
        summaries.append(dict(candidate=key, model=item["model"], feature_set=item["feature_set"],
                              weighting=item["weighting"], partition="test", **metrics(y_test, score, threshold)))
    pd.concat(all_rows, ignore_index=True).to_csv(OUT / "test_predictions.csv", index=False, float_format="%.17g")
    pd.DataFrame(summaries).to_csv(OUT / "test_metrics.csv", index=False, float_format="%.17g")
    compare_b(pd.DataFrame(summaries))
    print("PASS: evaluated 12 frozen models on A's test split")


def compare_b(c_metrics):
    paths = list((HERE.parent / "Group19_B" / "results" / "final_test").glob("*/metrics.csv"))
    if len(paths) != 1:
        print("B comparison skipped: expected one final B metrics file")
        return
    b = pd.read_csv(paths[0])
    b = b[b["feature_set"].isin(["raw", "engineered"])].copy()
    b["weighting"] = "not_applicable"
    b["source"] = "B"
    c = c_metrics.copy()
    c["source"] = "C"
    columns = ["source", "model", "feature_set", "weighting", "candidate", "average_precision",
               "roc_auc", "precision", "recall", "f1", "tp", "fp", "tn", "fn", "threshold"]
    pd.concat([b[columns], c[columns]], ignore_index=True).to_csv(OUT / "bc_comparison.csv", index=False)


def importance():
    """Validation-only permutation AP drop for each selected family/feature-set winner."""
    manifest = json.loads((OUT / "freeze_manifest.json").read_text(encoding="utf-8"))
    if manifest["models_sha256"] != sha256(OUT / "frozen_models.joblib"):
        raise ValueError("Frozen model file changed")
    fitted = joblib.load(OUT / "frozen_models.joblib")
    table = pd.read_csv(OUT / "validation_metrics.csv")
    rows = []
    for (name, feature_set), group in table.groupby(["model", "feature_set"], sort=False):
        winner = group.sort_values("average_precision", ascending=False, kind="stable").iloc[0]
        x_val, y_val, _ = load_split("validation", engineered=feature_set == "engineered")
        result = permutation_importance(fitted[winner["candidate"]], x_val, y_val,
                                        scoring="average_precision", n_repeats=3, random_state=42, n_jobs=1)
        for column, mean, std in zip(x_val.columns, result.importances_mean, result.importances_std):
            rows.append(dict(candidate=winner["candidate"], model=name, feature_set=feature_set,
                             weighting=winner["weighting"], feature=column,
                             ap_drop_mean=mean, ap_drop_std=std))
    pd.DataFrame(rows).to_csv(OUT / "permutation_importance.csv", index=False)
    print("PASS: validation-only permutation importance saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["tune", "evaluate-test", "importance"])
    command = parser.parse_args().command
    {"tune": tune, "evaluate-test": evaluate_test, "importance": importance}[command]()
