"""Shared anomaly scoring and validation-run artifacts."""

import hashlib
import json
import platform
import subprocess
from importlib.metadata import version

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor


MODELS = {"IsolationForest": IsolationForest, "LocalOutlierFactor": LocalOutlierFactor}


def anomaly_scores(model, features):
    return pd.Series(-model.score_samples(features.to_numpy()), index=features.index, name="anomaly_score")


def alarms_at_threshold(scores, threshold):
    return scores.ge(threshold)


def ranking_metrics(labels, scores):
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_validation_run(root, config, scores, candidates, fitted, staged):
    """Publish a complete run together so readers cannot mix scores and fitted models."""
    staged.mkdir()
    scores.to_csv(staged / "scores.csv", index=True, index_label="row_index", lineterminator="\n")
    joblib.dump(fitted, staged / "fits.joblib", compress=3)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root.parent, capture_output=True, text=True, check=True
    ).stdout.strip()
    sources = ["Group19_A/pipeline.py", "Group19_B/config.json", "Group19_B/run.py", "Group19_B/scoring.py"]
    log = {
        "git_head": commit,
        "source_sha256": {name: file_digest(root.parent / name) for name in sources},
        "environment": {"python": platform.python_version(), **{
            name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")
        }},
        "baseline": config["baseline"],
        "fit_partition": "train",
        "score_partition": "validation",
        "score": config["binary_alarm"]["score"],
        "score_direction": "higher_is_more_anomalous",
        "alarm_rule": config["binary_alarm"]["predict_positive_when"],
        "candidates": candidates,
        "artifact_sha256": {name: file_digest(staged / name) for name in ("scores.csv", "fits.joblib")},
    }
    manifest = staged / "manifest.json"
    manifest.write_bytes((json.dumps(log, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    destination = root / "results" / "validation" / file_digest(manifest)
    destination.mkdir(parents=True, exist_ok=True)
    # The manifest is the completion marker. Concurrent writers to this
    # content-addressed path publish identical bytes, one file at a time.
    for name in ("scores.csv", "fits.joblib", "manifest.json"):
        (staged / name).replace(destination / name)
    return destination
