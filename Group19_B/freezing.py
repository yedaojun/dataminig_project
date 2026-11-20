"""Select, calibrate and verify workstream B models using saved validation scores."""

import json
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from Group19_A import pipeline
from Group19_B import scoring


def best_f1_threshold(labels, scores):
    """Scan distinct observed thresholds, descending. Compare F1 as integer ratios."""
    ranked = pd.DataFrame({"score": scores, "positive": labels}).groupby("score")["positive"]
    groups = ranked.agg(["sum", "size"]).sort_index(ascending=False)
    positives = int(labels.sum())
    best_tp, best_denominator = -1, 1
    for threshold, (tp, alarms) in groups.cumsum().iterrows():
        tp = int(tp)
        denominator = positives + int(alarms)  # F1 = 2*TP / (positives + alarms)
        if tp * best_denominator > best_tp * denominator:
            best_threshold = float(threshold)
            best_tp, best_denominator = tp, denominator
    return best_threshold


def alarm_metrics(labels, alarms):
    tn, fp, fn, tp = map(int, confusion_matrix(labels, alarms, labels=[0, 1]).ravel())
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=tp / max(tp + fp, 1),
                recall=tp / max(tp + fn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1))


def read_manifest(directory, digest):
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("Invalid run fingerprint.")
    folder = directory / digest
    path = folder / "manifest.json"
    if scoring.file_digest(path) != digest:
        raise ValueError(f"Manifest fingerprint mismatch: {path}")
    manifest = json.loads(path.read_bytes())
    for name, expected in manifest["artifact_sha256"].items():
        if scoring.file_digest(folder / name) != expected:
            raise ValueError(f"Artifact fingerprint mismatch: {name}")
    return folder, manifest


def frozen_scores(features, record, fitted):
    if list(features.columns) != record["input_columns"]:
        raise ValueError("Frozen input column order differs from the supplied features.")
    matrix = fitted["preprocessor"].transform(features)
    if list(matrix.columns) != record["feature_names"] or not matrix.index.equals(features.index):
        raise ValueError("Frozen transformed column or row order mismatch.")
    return scoring.anomaly_scores(fitted["model"], matrix)


def load_frozen(root):
    directory = root / "results" / "frozen"
    pointer = directory / "CURRENT"
    if not pointer.is_file():
        raise ValueError("Missing frozen record. Run freeze before test evaluation.")
    folder, manifest = read_manifest(directory, pointer.read_text().strip())
    baseline = manifest["baseline"]
    if pipeline.fingerprint() != baseline["data_sha256"] or pipeline.split_fingerprint() != baseline["indices_sha256"]:
        raise ValueError("Frozen data or split fingerprint mismatch.")
    if scoring.file_digest(root / "config.json") != manifest["config_sha256"]:
        raise ValueError("Current protocol differs from the frozen configuration.")
    for record in manifest["selected"].values():
        expected = pipeline.INPUTS + (pipeline.ENGINEERED if record["engineered"] else [])
        if expected != record["input_columns"]:
            raise ValueError("Current pipeline input column order differs from the frozen schema.")
    return manifest, joblib.load(folder / "models.joblib")


def freeze(root):
    cfg = json.loads((root / "config.json").read_bytes())
    results = root / "results"
    table_path = results / "candidates_round2.csv"
    table = pd.read_csv(table_path, float_precision="round_trip").set_index("candidate", verify_integrity=True)
    run_ids = table["run_id"].unique()
    if len(run_ids) != 1:
        raise ValueError("Candidate table must identify one validation run.")
    source, log = read_manifest(results / "validation", run_ids[0])
    config_hash = scoring.file_digest(root / "config.json")
    if log["source_sha256"]["Group19_B/config.json"] != config_hash or log["baseline"] != cfg["baseline"]:
        raise ValueError("Validation run does not match the accepted protocol.")
    scores = pd.read_csv(source / "scores.csv", float_precision="round_trip").set_index("row_index", verify_integrity=True)
    labels = scores.pop(cfg["target"]["column"])
    if set(scores) != set(table.index) or set(scores) != set(log["candidates"]):
        raise ValueError("Candidate IDs differ across validation artifacts.")
    if not np.isfinite(scores.to_numpy()).all() or labels.value_counts().to_dict() != {0: 1449, 1: 51}:
        raise ValueError("Invalid validation scores or label counts.")
    for cid in scores:
        metrics = scoring.ranking_metrics(labels, scores[cid])
        if metrics != log["candidates"][cid]["validation_metrics"] or metrics != table.loc[cid, list(metrics)].to_dict():
            raise ValueError(f"Validation ranking metrics differ for {cid}.")

    fits = joblib.load(source / "fits.joblib")
    selected, objects, summary, predictions = {}, {}, [], []
    validation = {}
    for feature_set, settings in cfg["feature_sets"].items():
        features, original_labels, _ = pipeline.load_split("validation", engineered=settings["engineered"])
        if not original_labels.equals(labels):
            raise ValueError("Validation row IDs or labels differ from the frozen split.")
        validation[feature_set] = features
        for model, grid in cfg["models"].items():
            ids = [f"{feature_set}_{model}_{i}" for i in range(1, len(grid["candidates"]) + 1)]
            # max returns the first item on an exact tie, preserving the protocol's grid order.
            cid = max(ids, key=lambda key: table.at[key, "average_precision"])
            info = log["candidates"][cid]
            threshold = best_f1_threshold(labels, scores[cid])
            alarms = scoring.alarms_at_threshold(scores[cid], threshold)
            metrics = alarm_metrics(labels, alarms)
            selected[cid] = dict(info, engineered=settings["engineered"], input_columns=list(features),
                                 threshold=threshold, validation_alarm_metrics=metrics)
            objects[cid] = dict(model=fits["models"][cid], preprocessor=fits["preprocessors"][info["preprocessor_key"]])
            summary.append(dict(partition="validation", candidate=cid, feature_set=feature_set, model=model,
                                threshold=threshold, **info["validation_metrics"], **metrics))
            predictions.append(pd.DataFrame({"row_index": labels.index, "candidate": cid, "y_true": labels,
                                             "score": scores[cid], "threshold": threshold, "alarm": alarms.astype(int)}))

    directory = results / "frozen"
    directory.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=directory) as temp:
        staged = Path(temp)
        joblib.dump(objects, staged / "models.joblib", compress=3)
        pd.DataFrame(summary).to_csv(staged / "metrics.csv", index=False, lineterminator="\n")
        pd.concat(predictions).to_csv(staged / "predictions.csv", index=False, lineterminator="\n")
        restored = joblib.load(staged / "models.joblib")
        for cid, record in selected.items():
            replay = frozen_scores(validation[record["feature_set"]], record, restored[cid])
            if not replay.equals(scores[cid]) or not scoring.alarms_at_threshold(replay, record["threshold"]).equals(
                scoring.alarms_at_threshold(scores[cid], record["threshold"])
            ):
                raise ValueError(f"Frozen validation replay differs for {cid}.")
        manifest = {
            "validation_run": source.name, "candidate_table_sha256": scoring.file_digest(table_path),
            "config_sha256": config_hash, "baseline": log["baseline"], "fit_environment": log["environment"],
            "fit_git_head": log["git_head"], "fit_source_sha256": log["source_sha256"],
            "freeze_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "freeze_source_sha256": {name: scoring.file_digest(root / name) for name in ("run.py", "scoring.py", "freezing.py")},
            "selection": cfg["selection"], "binary_alarm": cfg["binary_alarm"], "score_direction": log["score_direction"],
            "reload_tolerance": {"rtol": 0, "atol": 0}, "selected": selected,
            "artifact_sha256": {p.name: scoring.file_digest(p) for p in sorted(staged.iterdir())},
        }
        path = staged / "manifest.json"
        path.write_bytes((json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode())
        digest = scoring.file_digest(path)
        destination = directory / digest
        destination.mkdir(exist_ok=True)
        for name in (*manifest["artifact_sha256"], "manifest.json"):
            (staged / name).replace(destination / name)
        pointer = staged / "CURRENT"
        pointer.write_text(digest + "\n", encoding="ascii")
        pointer.replace(directory / "CURRENT")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"PASS: saved and replayed four frozen validation models in {destination}")
