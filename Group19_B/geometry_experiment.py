"""Priority 4: one LOF distance ablation, removing redundant process temperature."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from Group19_A import pipeline
from Group19_B import freezing, scoring
from Group19_B.fusion_experiment import LOF_ID
from Group19_B.heat_experiment import BASELINE, ROOT, budget_rows
from Group19_B.run import check_alignment
from Group19_B.wear_experiment import FUSION_RUN


DROP = "Process temperature [K]"
TEMPS = ["Air temperature [K]", DROP, "temperature_difference"]


def neighborhood(model, xt, xv, train, queries):
    """Describe the actual LOF training neighbors of each validation query."""
    query = xv.loc[queries.index]
    positions = model.kneighbors(query.to_numpy(), return_distance=False)
    spread = (xt.to_numpy()[positions] - query.to_numpy()[:, None, :]) ** 2
    temp = xt.columns.isin(TEMPS)
    summary = pd.DataFrame(index=queries.index)
    summary["temperature_distance_share"] = spread[:, :, temp].sum(axis=(1, 2)) / spread.sum(axis=(1, 2))
    summary["same_type_share"] = (train.Type.to_numpy()[positions] == queries.Type.to_numpy()[:, None]).mean(axis=1)
    for column, key in [("temperature_difference", "delta_k"), ("Rotational speed [rpm]", "rpm"), ("wear_load", "wear_load")]:
        difference = train[column].to_numpy()[positions] - queries[column].to_numpy()[:, None]
        summary["mean_abs_neighbor_gap_" + key] = np.abs(difference).mean(axis=1)
    return summary, train.index.to_numpy()[positions]


def run():
    results = ROOT / "results"
    frozen, manifest = freezing.read_manifest(results / "frozen", BASELINE)
    fusion_dir, fusion_manifest = freezing.read_manifest(results / "fusion_priority2", FUSION_RUN)
    baseline = manifest["baseline"]
    if (fusion_manifest["baseline"] != BASELINE or pipeline.fingerprint() != baseline["data_sha256"]
            or pipeline.split_fingerprint() != baseline["indices_sha256"]
            or scoring.file_digest(ROOT / "config.json") != manifest["config_sha256"]):
        raise ValueError("Input data or protocol differs from the retained baseline.")
    train, _, _ = pipeline.load_split("train", engineered=True)
    val, y, modes = pipeline.load_split("validation", engineered=True)
    if not val.index.equals(modes.index) or not train.index.intersection(val.index).empty:
        raise ValueError("Development partitions or mode indices do not align.")
    prior = pd.read_csv(frozen / "predictions.csv", float_precision="round_trip")
    fusion = pd.read_csv(fusion_dir / "predictions.csv", index_col="row_index", float_precision="round_trip")
    if not fusion.y_true.equals(y):
        raise ValueError("Fusion validation rows differ.")
    info = manifest["selected"][LOF_ID]
    fitted = joblib.load(frozen / "models.joblib")[LOF_ID]
    xt, xv = (fitted["preprocessor"].transform(x) for x in (train, val))
    full = prior.loc[prior.candidate.eq(LOF_ID)].set_index("row_index", verify_integrity=True)
    if not full.y_true.equals(y) or not scoring.anomaly_scores(fitted["model"], xv).equals(full.score):
        raise ValueError("Engineering baseline replay differs.")
    reduced_t, reduced_v = xt.drop(columns=DROP), xv.drop(columns=DROP)
    model = scoring.MODELS["LocalOutlierFactor"](**info["params"]).fit(reduced_t.to_numpy())
    candidate = scoring.anomaly_scores(model, reduced_v)
    threshold = freezing.best_f1_threshold(y, candidate)
    scores = {"engineering_LOF": full.score, "drop_process_temperature": candidate, "fusion": fusion.fusion}
    thresholds = {"engineering_LOF": info["threshold"], "drop_process_temperature": threshold,
                  "fusion": pd.read_csv(fusion_dir / "metrics.csv", float_precision="round_trip").set_index("channel").at["fusion", "threshold"]}
    for cid, record in manifest["selected"].items():
        if record["feature_set"] == "raw":
            block = prior.loc[prior.candidate.eq(cid)].set_index("row_index", verify_integrity=True)
            if not block.y_true.equals(y):
                raise ValueError("Raw main comparison labels differ.")
            key = "raw_" + record["model"]
            scores[key], thresholds[key] = block.score, record["threshold"]

    masks = {"tp": y.eq(1), "fp": y.eq(0), "hdf": y.eq(1) & modes.HDF.eq(1),
             "pwf": y.eq(1) & modes.PWF.eq(1), "osf": y.eq(1) & modes.OSF.eq(1)}
    masks["L_osf"] = masks["osf"] & val.Type.eq("L")
    queries = val.loc[masks["L_osf"] & ~full.alarm.astype(bool)]
    before, before_ids = neighborhood(fitted["model"], xt, xv, train, queries)
    after, after_ids = neighborhood(model, reduced_t, reduced_v, train, queries)
    neighbors = before.add_prefix("before_").join(after.add_prefix("after_"))
    neighbors["shared_neighbors"] = [len(set(a) & set(b)) for a, b in zip(before_ids, after_ids)]
    neighbors["before_rank"] = full.score.rank(ascending=False).loc[queries.index]
    neighbors["after_rank"] = candidate.rank(ascending=False).loc[queries.index]
    neighbors["recovered"] = candidate.loc[queries.index].ge(threshold)
    pred = pd.concat([val.Type, y.rename("y_true"), modes], axis=1)
    metrics, budgets, changes = [], [], []
    operating = {}
    for name, s in scores.items():
        check_alignment(val, y, s)
        alarm = scoring.alarms_at_threshold(s, thresholds[name])
        pred[name + "_score"], pred[name + "_alarm"] = s, alarm.astype(int)
        operating[name] = {"calibrated": alarm, "top41_through_ties": s.ge(s.nlargest(41).min())}
        metrics.append(dict(candidate=name, threshold=thresholds[name], **scoring.ranking_metrics(y, s),
                            **freezing.alarm_metrics(y, alarm),
                            **{k: int((alarm & mask).sum()) for k, mask in masks.items() if k not in ("tp", "fp")}))
        budgets.extend(budget_rows(name, s, y, masks["hdf"], masks["pwf"]))
    for point, new in operating["drop_process_temperature"].items():
        old = operating["engineering_LOF"][point]
        for group, mask in masks.items():
            changes.append(dict(operating_point=point, group=group, before=int((old & mask).sum()),
                                after=int((new & mask).sum()), gained=int((new & ~old & mask).sum()),
                                lost=int((old & ~new & mask).sum())))
    correlations = train[pipeline.NUM + pipeline.ENGINEERED].corr()
    residual = (train[DROP] - (train[TEMPS[0]] + train[TEMPS[2]])).abs().max()
    output = results / "geometry_priority4"
    output.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=output) as tmp:
        stage = Path(tmp)
        for name, table in (("metrics", metrics), ("budgets", budgets), ("changes", changes)):
            pd.DataFrame(table).to_csv(stage / (name + ".csv"), index=False, lineterminator="\n")
        for name, table, index in (("predictions", pred, "row_index"), ("neighbors", neighbors, "row_index"),
                                   ("train_correlations", correlations, "feature")):
            table.to_csv(stage / (name + ".csv"), index_label=index, lineterminator="\n")
        log = dict(frozen=BASELINE, retained_fusion=FUSION_RUN, paired_candidate=LOF_ID,
                   data_sha256=baseline["data_sha256"], indices_sha256=baseline["indices_sha256"],
                   scope="engineering-only ablation; raw features remain the primary B/C comparison",
                   dropped_column=DROP, retained_columns=list(reduced_t), params=info["params"],
                   train_max_abs_process_reconstruction_error=float(residual),
                   source_sha256=scoring.file_digest(Path(__file__)),
                   artifact_sha256={p.name: scoring.file_digest(p) for p in sorted(stage.iterdir())})
        path = stage / "manifest.json"
        path.write_bytes((json.dumps(log, indent=2) + "\n").encode())
        dest = output / scoring.file_digest(path)
        dest.mkdir(exist_ok=True)
        for name in (*log["artifact_sha256"], "manifest.json"):
            (stage / name).replace(dest / name)
    print(pd.DataFrame(metrics).to_string(index=False))
    print(pd.DataFrame(changes).to_string(index=False))
    print(neighbors.mean().to_string())
    print(f"Saved {dest}")


if __name__ == "__main__":
    run()
