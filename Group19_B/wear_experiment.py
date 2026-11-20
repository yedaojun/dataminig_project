"""Priority 3: add one training-only, type-relative wear-load feature."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from Group19_A import pipeline
from Group19_B import freezing, scoring
from Group19_B.fusion_experiment import percentile
from Group19_B.heat_experiment import BASELINE, ROOT
from Group19_B.run import check_alignment


FUSION_RUN = "4db6488900fe61a295e73994d50ce72eaa8b968f99767fb731ec3f4637babc4b"
MIN_TYPE_SIZE = 100  # At least 1-percentile-point ECDF resolution.


def relative_load(train, sample):
    # Torque is recorded in tenths of Nm and tool wear in whole minutes.
    reference = train["wear_load"].round(1)
    values = sample["wear_load"].round(1)
    ranks = percentile(reference, values)
    for kind, group in reference.groupby(train["Type"]):
        if len(group) >= MIN_TYPE_SIZE:
            mask = sample["Type"].eq(kind)
            ranks.loc[mask] = percentile(group, values.loc[mask])
    return ranks.rename("wear_load_type_percentile")


def run():
    results = ROOT / "results"
    frozen, log = freezing.read_manifest(results / "frozen", BASELINE)
    prior, fusion_log = freezing.read_manifest(results / "fusion_priority2", FUSION_RUN)
    if (fusion_log["baseline"] != BASELINE or scoring.file_digest(ROOT / "config.json") != log["config_sha256"]
            or pipeline.fingerprint() != log["baseline"]["data_sha256"]
            or pipeline.split_fingerprint() != log["baseline"]["indices_sha256"]):
        raise ValueError("Data, split or protocol differs from the retained baselines.")

    train, _, _ = pipeline.load_split("train", engineered=True)
    val, y, modes = pipeline.load_split("validation", engineered=True)
    if (len(train), len(val)) != (7000, 1500) or not train.index.intersection(val.index).empty:
        raise ValueError("Unexpected development partitions.")
    train_rank, val_rank = relative_load(train, train), relative_load(train, val)
    prior_scores = pd.read_csv(prior / "predictions.csv", index_col="row_index", float_precision="round_trip")
    if not y.equals(prior_scores.y_true) or not y.index.equals(modes.index):
        raise ValueError("Retained fusion validation rows differ.")
    frozen_rows = pd.read_csv(frozen / "predictions.csv", float_precision="round_trip")
    fusion_metrics = pd.read_csv(prior / "metrics.csv", float_precision="round_trip").set_index("channel")
    scores = {"fusion": prior_scores.fusion}
    thresholds = {"fusion": fusion_metrics.at["fusion", "threshold"]}
    controls = {}
    params = {}
    for cid, record in log["selected"].items():
        if not record["engineered"]:
            continue
        model = record["model"]
        block = frozen_rows.loc[frozen_rows.candidate.eq(cid)].set_index("row_index", verify_integrity=True)
        if not block.y_true.equals(y):
            raise ValueError("Paired baseline validation rows differ.")
        base = "engineering_" + model
        scores[base] = block.score
        thresholds[base] = record["threshold"]
        pre = pipeline.make_preprocessor(engineered=True, scale=record["scale_numeric"]).set_output(transform="pandas")
        x_train = pre.fit_transform(train)
        x_val = pre.transform(val)
        extra_train, extra_val = train_rank.to_frame(), val_rank.to_frame()
        if record["scale_numeric"]:
            scaler = StandardScaler().set_output(transform="pandas")
            extra_train = scaler.fit_transform(extra_train)
            extra_val = scaler.transform(extra_val)
        x_train, x_val = x_train.join(extra_train), x_val.join(extra_val)
        candidate = "type_load_" + model
        params[candidate] = record["params"]
        estimator = scoring.MODELS[model](**record["params"]).fit(x_train.to_numpy())
        scores[candidate] = scoring.anomaly_scores(estimator, x_val)
        controls[candidate] = base

    pred = pd.concat([y.rename("y_true"), val[["Type", "wear_load"]], val_rank, modes], axis=1)
    masks = dict(tp=y.eq(1), fp=y.eq(0), osf=y.eq(1) & modes.OSF.eq(1),
                 hdf=y.eq(1) & modes.HDF.eq(1), pwf=y.eq(1) & modes.PWF.eq(1))
    metric_rows, type_rows, paired_rows = [], [], []
    alarms = {}
    train_counts = train.Type.value_counts().to_dict()
    for name, values in scores.items():
        check_alignment(val, y, values)
        threshold = thresholds[name] if name in thresholds else freezing.best_f1_threshold(y, values)
        alarm = scoring.alarms_at_threshold(values, threshold)
        cut = values.nlargest(41).min()
        above, tied = values.gt(cut), values.eq(cut)
        slots = 41 - int(above.sum())
        alarms[name] = {"calibrated": alarm, "top41_through_ties": above | tied}
        pred[name + "_score"] = values
        pred[name + "_alarm"] = alarm.astype(int)
        row = dict(candidate=name, threshold=threshold, **scoring.ranking_metrics(y, values),
                   **freezing.alarm_metrics(y, alarm), budget=41, cutoff=cut,
                   above=int(above.sum()), through_ties=int((above | tied).sum()))
        for key, mask in masks.items():
            row[key + "_at41_min"] = int((above & mask).sum()) + max(0, slots - int((tied & ~mask).sum()))
            row[key + "_at41_max"] = int((above & mask).sum()) + min(slots, int((tied & mask).sum()))
            if key not in ("tp", "fp"):
                row[key] = int((alarm & mask).sum())
        metric_rows.append(row)
        for point, called in alarms[name].items():
            for kind, indices in val.groupby("Type").groups.items():
                type_rows.append(dict(candidate=name, operating_point=point, product_type=kind,
                                      train_rows=train_counts.get(kind, 0), validation_rows=len(indices),
                                      normals=int(masks["fp"].loc[indices].sum()), osf_support=int(masks["osf"].loc[indices].sum()),
                                      osf_detected=int((called & masks["osf"]).loc[indices].sum()),
                                      false_alarms=int((called & masks["fp"]).loc[indices].sum())))
    for candidate, base in controls.items():
        for comparator in (base, "fusion"):
            for point, new in alarms[candidate].items():
                old = alarms[comparator][point]
                row = dict(candidate=candidate, comparator=comparator, operating_point=point)
                for key, mask in masks.items():
                    row[key + "_new"] = int((new & ~old & mask).sum())
                    row[key + "_lost"] = int((~new & old & mask).sum())
                    row[key + "_kept"] = int((new & old & mask).sum())
                paired_rows.append(row)

    output = results / "wear_priority3"
    output.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=output) as tmp:
        staged = Path(tmp)
        pred.to_csv(staged / "predictions.csv", index_label="row_index", lineterminator="\n")
        for name, rows in (("metrics", metric_rows), ("by_type", type_rows), ("changes", paired_rows)):
            pd.DataFrame(rows).to_csv(staged / (name + ".csv"), index=False, lineterminator="\n")
        manifest = dict(retained_best=FUSION_RUN, paired_baseline=BASELINE, model_params=params,
                        training_support=train_counts, fallback_below=MIN_TYPE_SIZE,
                        feature="ECDF within product Type of training wear_load rounded to 0.1 minNm; ties use <=",
                        data_sha256=pipeline.fingerprint(), indices_sha256=pipeline.split_fingerprint(),
                        source_sha256=scoring.file_digest(Path(__file__)),
                        artifact_sha256={p.name: scoring.file_digest(p) for p in sorted(staged.iterdir())})
        path = staged / "manifest.json"
        path.write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
        dest = output / scoring.file_digest(path)
        dest.mkdir(exist_ok=True)
        for name in (*manifest["artifact_sha256"], "manifest.json"):
            (staged / name).replace(dest / name)
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(pd.DataFrame(type_rows).query("operating_point == 'calibrated'").to_string(index=False))
    print(pd.DataFrame(paired_rows).to_string(index=False))
    print(f"Saved {dest}")


if __name__ == "__main__":
    run()
