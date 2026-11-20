"""Priority 5: characterize high-scoring normal rows and test one torque residual."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from Group19_A import pipeline
from Group19_B import freezing, scoring
from Group19_B.fusion_experiment import LOF_ID
from Group19_B.heat_experiment import BASELINE, ROOT, budget_rows
from Group19_B.run import check_alignment
from Group19_B.wear_experiment import FUSION_RUN


RPM = "Rotational speed [rpm]"
TORQUE = "Torque [Nm]"
FIELDS = {"torque": TORQUE, "power": "mechanical_power", "wear": "Tool wear [min]"}


def reference_bins(train, speeds):
    bins = train[RPM] // 100
    rows = []
    for band in sorted(set(speeds // 100)):
        distance = (bins - band).abs()
        radius = distance.nsmallest(30).max()
        pool = train.loc[distance <= radius]
        row = dict(band=band, radius=radius, train_n=len(pool), median_torque=pool[TORQUE].median())
        for key, column in FIELDS.items():
            row[key + "_q10"], row[key + "_q90"] = pool[column].quantile([.1, .9])
        rows.append(row)
    return pd.DataFrame(rows).set_index("band")


def run():
    frozen, log = freezing.read_manifest(ROOT / "results/frozen", BASELINE)
    fusion_dir, fusion_log = freezing.read_manifest(ROOT / "results/fusion_priority2", FUSION_RUN)
    if (fusion_log["baseline"] != BASELINE or pipeline.fingerprint() != log["baseline"]["data_sha256"]
            or pipeline.split_fingerprint() != log["baseline"]["indices_sha256"]
            or scoring.file_digest(ROOT / "config.json") != log["config_sha256"]):
        raise ValueError("Baseline or development data mismatch.")
    train, _, _ = pipeline.load_split("train", engineered=True)
    val, y, modes = pipeline.load_split("validation", engineered=True)
    prior = pd.read_csv(fusion_dir / "predictions.csv", index_col="row_index", float_precision="round_trip")
    if not prior.y_true.equals(y) or not modes.index.equals(y.index) or not train.index.intersection(val.index).empty:
        raise ValueError("Validation row or label mismatch.")
    # Query speeds determine which bins need lookup. All reference statistics use training rows only.
    bins = reference_bins(train, pd.concat([train[RPM], val[RPM]]))
    train_band, val_band = train[RPM] // 100, val[RPM] // 100
    train_resid = train[TORQUE] - train_band.map(bins.median_torque)
    val_resid = val[TORQUE] - val_band.map(bins.median_torque)
    fitted = joblib.load(frozen / "models.joblib")[LOF_ID]
    params = log["selected"][LOF_ID]["params"]
    xt, xv = (fitted["preprocessor"].transform(x) for x in (train, val))
    if not scoring.anomaly_scores(fitted["model"], xv).equals(prior.lof):
        raise ValueError("Engineering LOF replay differs from retained scores.")
    scaler = StandardScaler()
    xt["torque_residual"] = scaler.fit_transform(train_resid.to_frame()).ravel()
    xv["torque_residual"] = scaler.transform(val_resid.to_frame()).ravel()
    model = scoring.MODELS["LocalOutlierFactor"](**params).fit(xt.to_numpy())
    new_scores = scoring.anomaly_scores(model, xv)
    high_cut = train[RPM].quantile(.90)
    high = val[RPM].ge(high_cut)
    pred = pd.concat([y.rename("y_true"), modes, val[["Type", RPM, *FIELDS.values()]]], axis=1)
    pred["speed_band"], pred["high_speed"], pred["torque_residual"] = val_band, high, val_resid
    flags = []
    for key, column in FIELDS.items():
        field = key + "_central"
        pred[field] = val[column].between(val_band.map(bins[key + "_q10"]), val_band.map(bins[key + "_q90"]))
        flags.append(field)
    pred["all_central"] = pred[flags].all(axis=1)
    masks = dict(tp=y.eq(1), fp=y.eq(0), pwf=y.eq(1) & modes.PWF.eq(1),
                 hdf=y.eq(1) & modes.HDF.eq(1), osf=y.eq(1) & modes.OSF.eq(1),
                 high_speed_fp=high & y.eq(0))
    thresholds = pd.read_csv(fusion_dir / "metrics.csv", float_precision="round_trip").set_index("channel").threshold.to_dict()
    thresholds["residual"] = freezing.best_f1_threshold(y, new_scores)
    metrics, budgets, changes = [], [], []
    groups = {"all_normal": y.eq(0), "all_pwf": masks["pwf"]}
    operating = {}
    for name, values in {"lof": prior.lof, "fusion": prior.fusion, "residual": new_scores}.items():
        check_alignment(val, y, values)
        alarm = scoring.alarms_at_threshold(values, thresholds[name])
        if name != "residual" and not alarm.equals(prior[name + "_alarm"].astype(bool)):
            raise ValueError("Reference alerts differ.")
        operating[name] = {"calibrated": alarm, "top41_through_ties": values.ge(values.nlargest(41).min())}
        pred[name + "_score"], pred[name + "_alarm"] = values, alarm.astype(int)
        groups[name + "_fp"] = alarm & y.eq(0)
        metrics.append(dict(candidate=name, threshold=thresholds[name], **scoring.ranking_metrics(y, values),
                            **freezing.alarm_metrics(y, alarm), pwf=int((alarm & masks["pwf"]).sum())))
        budgets.extend(budget_rows(name, values, y, masks["hdf"], masks["pwf"]))
    groups["lof_fusion_shared_fp"] = groups["lof_fp"] & groups["fusion_fp"]
    conditions = []
    for name, mask in groups.items():
        subset = pred.loc[mask & high]
        conditions.append(dict(group=name, n=int(mask.sum()), median_rpm=val.loc[mask, RPM].median(),
                               high_n=len(subset), **{f: int(subset[f].sum()) for f in flags + ["all_central"]}))
    for name in ("lof", "fusion"):
        for point, new in operating["residual"].items():
            old = operating[name][point]
            for key, mask in masks.items():
                changes.append(dict(reference=name, operating_point=point, group=key, before=int((old & mask).sum()),
                                    after=int((new & mask).sum()), added=int((new & ~old & mask).sum()),
                                    removed=int((~new & old & mask).sum())))

    output = ROOT / "results/operating_priority5"
    output.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=output) as tmp:
        staged = Path(tmp)
        for name, rows in (("metrics", metrics), ("budgets", budgets), ("changes", changes), ("conditions", conditions)):
            pd.DataFrame(rows).to_csv(staged / (name + ".csv"), index=False, lineterminator="\n")
        bins.to_csv(staged / "train_bins.csv", lineterminator="\n")
        pred.to_csv(staged / "predictions.csv", index_label="row_index", lineterminator="\n")
        manifest = dict(paired_baseline=BASELINE, retained_fusion=FUSION_RUN, params=params,
                        feature="signed torque minus training median within rpm band",
                        bin_width_rpm=100, minimum_pool=30, high_speed_rpm=float(high_cut),
                        central_quantiles=[.1, .9], data_sha256=pipeline.fingerprint(),
                        indices_sha256=pipeline.split_fingerprint(), source_sha256=scoring.file_digest(Path(__file__)),
                        artifact_sha256={p.name: scoring.file_digest(p) for p in sorted(staged.iterdir())})
        path = staged / "manifest.json"
        path.write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
        dest = output / scoring.file_digest(path)
        dest.mkdir(exist_ok=True)
        for name in (*manifest["artifact_sha256"], "manifest.json"):
            (staged / name).replace(dest / name)
    print(pd.DataFrame(conditions).to_string(index=False))
    print(pd.DataFrame(metrics).to_string(index=False))
    print(pd.DataFrame(budgets).query("budget == 41").to_string(index=False))
    print(pd.DataFrame(changes).query("reference == 'lof'").to_string(index=False))
    print(f"Saved {dest}")


if __name__ == "__main__":
    run()
