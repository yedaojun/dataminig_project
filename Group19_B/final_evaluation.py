"""Evaluate the approved B candidates after exact validation replay."""

import json
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from Group19_A import pipeline
from Group19_B import freezing, scoring
from Group19_B.fusion_experiment import LOF_ID, percentile
from Group19_B.heat_experiment import BASELINE, joint_low
from Group19_B.run import check_alignment
from Group19_B.wear_experiment import FUSION_RUN


FUSION = "heat_lof_percentile_max"


def read_table(path):
    return pd.read_csv(path, float_precision="round_trip")


def predict(features, labels, records, fitted, train, oof, cutoffs):
    if list(features) != pipeline.INPUTS + pipeline.ENGINEERED:
        raise ValueError("Input schema differs from the frozen engineering group.")
    scores = {}
    for cid, rec in records.items():
        inputs = features if rec["engineered"] else features[pipeline.INPUTS]
        scores[cid] = freezing.frozen_scores(inputs, rec, fitted[cid])
    scores[FUSION] = np.maximum(percentile(oof.h, joint_low(train, features)),
                                percentile(oof.lof, scores[LOF_ID]))
    blocks = []
    for cid, values in scores.items():
        check_alignment(features, labels, values)
        rec = records.get(cid, {"model": FUSION, "feature_set": "heat_fusion"})
        threshold = cutoffs[cid]
        blocks.append(pd.DataFrame(dict(row_index=labels.index, candidate=cid, model=rec["model"],
                                        feature_set=rec["feature_set"], y_true=labels, score=values,
                                        threshold=threshold, alarm=scoring.alarms_at_threshold(values, threshold).astype(int))))
    return pd.concat(blocks, ignore_index=True)


def verify_replay(replay, frozen_folder, fusion_folder):
    baseline = read_table(frozen_folder / "predictions.csv")
    fusion = read_table(fusion_folder / "predictions.csv").rename(columns={"fusion": "score", "fusion_alarm": "alarm"})
    for cid, rows in replay.groupby("candidate", sort=False):
        reference = fusion if cid == FUSION else baseline.loc[baseline.candidate.eq(cid)]
        for col in ("row_index", "y_true", "score", "alarm"):
            if not np.array_equal(rows[col], reference[col]):
                raise ValueError(f"Frozen validation replay mismatch for {cid}/{col}.")


def export_metrics(staged):
    # Read the exact public predictions so every summary and plotted point has one source.
    predictions = read_table(staged / "predictions.csv")
    metrics, points = [], []
    for cid, rows in predictions.groupby("candidate", sort=False):
        y, score, alarm = rows.y_true, rows.score, rows.alarm
        if not np.array_equal(alarm, score.ge(rows.threshold)):
            raise ValueError(f"Published alarm mismatch for {cid}.")
        counts = freezing.alarm_metrics(y, alarm)
        if (counts["tp"] + counts["fn"], counts["tn"] + counts["fp"]) != (51, 1449):
            raise ValueError("Test confusion matrix has unexpected margins.")
        metrics.append(dict(partition="test", candidate=cid, model=rows.model.iloc[0],
                            feature_set=rows.feature_set.iloc[0], threshold=rows.threshold.iloc[0],
                            **scoring.ranking_metrics(y, score), **counts))
        precision, recall, _ = precision_recall_curve(y, score)
        fpr, tpr, _ = roc_curve(y, score, drop_intermediate=False)
        for curve, x, ordinate in (("PR", recall, precision), ("ROC", fpr, tpr)):
            points.append(pd.DataFrame(dict(candidate=cid, curve=curve, x=x, y=ordinate)))
    pd.DataFrame(metrics).to_csv(staged / "metrics.csv", index=False, lineterminator="\n")
    pd.concat(points).to_csv(staged / "curves.csv", index=False, lineterminator="\n")


def evaluate_test(root):
    results = root / "results"
    if (results / "frozen/CURRENT").read_text().strip() != BASELINE:
        raise ValueError("CURRENT differs from the baseline approved for final testing.")
    manifest, fitted = freezing.load_frozen(root)
    records, baseline = manifest["selected"], manifest["baseline"]
    frozen_folder = results / "frozen" / BASELINE
    fusion_folder, fusion = freezing.read_manifest(results / "fusion_priority2", FUSION_RUN)
    if (fusion["baseline"] != BASELINE or fusion["model"] != LOF_ID
            or any(fusion[key] != baseline[key] for key in ("data_sha256", "indices_sha256"))):
        raise ValueError("Fusion reference differs from the frozen baseline.")
    cutoffs = {cid: rec["threshold"] for cid, rec in records.items()}
    cutoffs[FUSION] = read_table(fusion_folder / "metrics.csv").set_index("channel").at["fusion", "threshold"]
    train, _, _ = pipeline.load_split("train", engineered=True)
    val, y_val, _ = pipeline.load_split("validation", engineered=True)
    oof = read_table(fusion_folder / "oof.csv").set_index("row_index", verify_integrity=True)
    if len(train) != 7000 or not oof.index.equals(train.index) or not np.isfinite(oof[["h", "lof"]]).all().all():
        raise ValueError("Out-of-fold reference rows differ from the frozen training set.")
    replay = predict(val, y_val, records, fitted, train, oof, cutoffs)
    verify_replay(replay, frozen_folder, fusion_folder)

    plot_python = sys.executable if find_spec("matplotlib") else sys._base_executable
    plot_version = subprocess.check_output([plot_python, "-B", "-c",
        "import matplotlib; print(matplotlib.__version__)"], text=True).strip()
    print("PASS: validation replay matched all five saved candidates; opening the test split.", flush=True)
    test, y_test, _ = pipeline.load_split("test", engineered=True, allow_test=True)
    with np.load((root.parent / baseline["manifest"]).with_name("indices.npz")) as split:
        if (not np.array_equal(test.index, split["test"]) or len(test) != 1500
                or len(test.index.intersection(train.index.union(val.index))) != 0
                or y_test.value_counts().to_dict() != {0: 1449, 1: 51}):
            raise ValueError("Test rows, partition overlap or class totals differ from the protocol.")
    predictions = predict(test, y_test, records, fitted, train, oof, cutoffs)
    directory = results / "final_test"
    directory.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=directory) as temp:
        staged = Path(temp) / "run"
        staged.mkdir()
        predictions.insert(0, "partition", "test")
        predictions.to_csv(staged / "predictions.csv", index=False, lineterminator="\n")
        export_metrics(staged)
        subprocess.run([plot_python, "-B", str(root / "plot_test_curves.py"), str(staged)], check=True)
        sources = [Path(__file__), root / "run.py", root / "plot_test_curves.py", root / "fusion_experiment.py",
                   root / "heat_experiment.py", root / "wear_experiment.py", root / "scoring.py",
                   root / "freezing.py", Path(pipeline.__file__)]
        log = dict(baseline_run=BASELINE, fusion_run=FUSION_RUN, data_sha256=baseline["data_sha256"],
                   indices_sha256=baseline["indices_sha256"], split="test", positive_label=1,
                   groups={"raw": "B/C primary", "engineered": "common engineered inputs", "heat_fusion": "B extension"},
                   validation_replay_rows=len(replay), thresholds=cutoffs, alarm_rule="score >= threshold",
                   plot_matplotlib=plot_version, fit_environment=manifest["fit_environment"],
                   source_sha256={p.relative_to(root.parent).as_posix(): scoring.file_digest(p) for p in sources},
                   artifact_sha256={p.name: scoring.file_digest(p) for p in sorted(staged.iterdir())})
        path = staged / "manifest.json"
        path.write_bytes((json.dumps(log, indent=2, allow_nan=False) + "\n").encode())
        digest = scoring.file_digest(path)
        destination = directory / digest
        if destination.exists():
            freezing.read_manifest(directory, digest)
        else:
            staged.rename(destination)
    print(read_table(destination / "metrics.csv").to_string(index=False))
    print(f"PASS: final test artifacts in {destination}")
