"""Priority 1: validation-only heat-feature ablation against the frozen engineering models."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from Group19_A import pipeline
from Group19_B import freezing, scoring
from Group19_B.run import check_alignment


ROOT = Path(__file__).resolve().parent
BASELINE = "38bfe87136caff9241c459e5061c6ec4e53236b7174f6f4598483a7909b30d71"
THERMAL = ["temperature_difference", "Rotational speed [rpm]"]


def joint_low(train, sample):
    percentiles = []
    for col in THERMAL:
        reference, values = train[col], sample[col]
        if col == "temperature_difference":
            # Temperatures are reported to 0.1 K. Preserve ties after floating-point subtraction.
            reference, values = reference.round(1), values.round(1)
        percentiles.append(np.searchsorted(np.sort(reference), values, side="right") / len(train))
    return pd.Series(1 - np.maximum(*percentiles), index=sample.index, name="joint_low")


def budget_rows(name, scores, labels, hdf, pwf):
    ranked = scores.sort_values(ascending=False)
    rows = []
    for k in (20, 41, 61, 101):
        cutoff = ranked.iloc[k - 1]
        above, tied = scores.gt(cutoff), scores.eq(cutoff)
        slots = k - int(above.sum())
        row = dict(candidate=name, budget=k, cutoff=cutoff, above=int(above.sum()),
                   through_ties=int((above | tied).sum()))
        for field, mask in (("tp", labels.eq(1)), ("hdf", hdf), ("pwf", pwf)):
            hits = int((mask & above).sum())
            positives_tied = int((mask & tied).sum())
            row[field + "_above"] = hits
            row[field + "_through_ties"] = hits + positives_tied
            row[field + "_min"] = hits + max(0, slots - int((~mask & tied).sum()))
            row[field + "_max"] = hits + min(slots, positives_tied)
        rows.append(row)
    return rows


def run():
    frozen_dir, manifest = freezing.read_manifest(ROOT / "results/frozen", BASELINE)
    if (pipeline.fingerprint() != manifest["baseline"]["data_sha256"]
            or pipeline.split_fingerprint() != manifest["baseline"]["indices_sha256"]
            or scoring.file_digest(ROOT / "config.json") != manifest["config_sha256"]):
        raise ValueError("Data, split or protocol differs from the approved baseline.")
    original = pd.read_csv(frozen_dir / "predictions.csv", float_precision="round_trip")
    train, _, _ = pipeline.load_split("train", engineered=True)
    val, labels, modes = pipeline.load_split("validation", engineered=True)
    if not val.index.equals(modes.index) or (len(train), len(val)) != (7000, 1500):
        raise ValueError("Development split mismatch.")
    hdf = labels.eq(1) & modes["HDF"].eq(1)
    pwf = labels.eq(1) & modes["PWF"].eq(1)
    h_train, h_val = joint_low(train, train), joint_low(train, val)
    scores_by_name = {"h_only": h_val}
    base_alarms, base_names, thresholds, params = {}, {}, {}, {}
    for cid, rec in manifest["selected"].items():
        if rec["feature_set"] != "engineered":
            continue
        model = rec["model"]
        base = "baseline_" + model
        block = original.loc[original.candidate.eq(cid)].set_index("row_index", verify_integrity=True)
        if not labels.equals(block.y_true):
            raise ValueError("Frozen validation labels or row order mismatch.")
        scores_by_name[base] = block.score
        base_alarms[base] = block.alarm.astype(bool)
        thresholds[base] = rec["threshold"]
        params[model] = rec["params"]
        base_names[base] = base

        thermal_train, thermal_val = train[THERMAL], val[THERMAL]
        h_col_train, h_col_val = h_train.to_frame(), h_val.to_frame()
        if rec["scale_numeric"]:
            scaler = StandardScaler().set_output(transform="pandas")
            thermal_train = scaler.fit_transform(thermal_train)
            thermal_val = scaler.transform(thermal_val)
            h_scaler = StandardScaler().set_output(transform="pandas")
            h_col_train = h_scaler.fit_transform(h_col_train)
            h_col_val = h_scaler.transform(h_col_val)
        pre = pipeline.make_preprocessor(engineered=True, scale=rec["scale_numeric"]).set_output(transform="pandas")
        full_train = pre.fit_transform(train).join(h_col_train)
        full_val = pre.transform(val).join(h_col_val)
        for design, x_train, x_val in (("thermal", thermal_train, thermal_val),
                                       ("engineered_h", full_train, full_val)):
            name = f"{design}_{model}"
            estimator = scoring.MODELS[model](**rec["params"])
            estimator.fit(x_train.to_numpy())
            scores_by_name[name] = scoring.anomaly_scores(estimator, x_val)
            base_names[name] = base

    predictions = pd.DataFrame({"y_true": labels, "HDF": modes.HDF, "PWF": modes.PWF}, index=val.index)
    lof = base_alarms["baseline_LocalOutlierFactor"]
    summary, budgets = [], []
    for name, scores in scores_by_name.items():
        check_alignment(val, labels, scores)
        predictions[name + "_score"] = scores
        budgets.extend(budget_rows(name, scores, labels, hdf, pwf))
        row = dict(candidate=name, **scoring.ranking_metrics(labels, scores))
        if name != "h_only":
            threshold = thresholds[name] if name in thresholds else freezing.best_f1_threshold(labels, scores)
            alarms = scoring.alarms_at_threshold(scores, threshold)
            base = base_alarms[base_names[name]]
            if name in thresholds and not alarms.equals(base):
                raise ValueError("Frozen baseline alarms differ.")
            predictions[name + "_alarm"] = alarms.astype(int)
            row.update(threshold=threshold, **freezing.alarm_metrics(labels, alarms),
                       paired_baseline=base_names[name], hdf_detected=int((alarms & hdf).sum()),
                       pwf_detected=int((alarms & pwf).sum()), hdf_recovered_vs_lof=int((alarms & ~lof & hdf).sum()),
                       pwf_retained_vs_lof=int((alarms & lof & pwf).sum()))
            for key, mask in (("tp", labels.eq(1)), ("fp", labels.eq(0))):
                row[key + "_added"] = int((alarms & ~base & mask).sum())
                row[key + "_removed"] = int((~alarms & base & mask).sum())
        summary.append(row)

    output = ROOT / "results/heat_priority1"
    output.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=output) as tmp:
        staged = Path(tmp)
        predictions.to_csv(staged / "predictions.csv", index_label="row_index", lineterminator="\n")
        pd.DataFrame(summary).to_csv(staged / "metrics.csv", index=False, lineterminator="\n")
        pd.DataFrame(budgets).to_csv(staged / "budgets.csv", index=False, lineterminator="\n")
        log = dict(baseline=BASELINE, model_params=params, training_rows=len(train), validation_rows=len(val),
                   data_sha256=pipeline.fingerprint(), indices_sha256=pipeline.split_fingerprint(),
                   source_sha256=scoring.file_digest(Path(__file__)), columns_thermal=THERMAL,
                   joint_low="1 - max(train ECDF(delta_T rounded to 0.1 K), train ECDF(rpm)); ties use <=",
                   artifact_sha256={p.name: scoring.file_digest(p) for p in sorted(staged.iterdir())})
        path = staged / "manifest.json"
        path.write_bytes((json.dumps(log, indent=2) + "\n").encode())
        dest = output / scoring.file_digest(path)
        dest.mkdir(exist_ok=True)
        for name in (*log["artifact_sha256"], "manifest.json"):
            (staged / name).replace(dest / name)
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(budgets).query("budget == 41").to_string(index=False))
    print(f"Saved {dest}")


if __name__ == "__main__":
    run()
