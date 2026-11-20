"""Priority 2: out-of-fold percentile maximum of heat risk and frozen engineering LOF."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from Group19_A import pipeline
from Group19_B import freezing, scoring
from Group19_B.heat_experiment import BASELINE, ROOT, budget_rows, joint_low
from Group19_B.run import check_alignment


HEAT_RUN = "44d066085359f4b804fb0394c34a02e7561940058768d0f60813b072cf04f121"
LOF_ID = "engineered_LocalOutlierFactor_7"


def percentile(reference, values):
    return pd.Series(np.searchsorted(np.sort(reference), values, side="right") / len(reference),
                     index=values.index)


def build_reference(train, validation, full_scores, params):
    oof = pd.DataFrame(index=train.index, columns=["h", "lof"], dtype=float)
    oof["fold"] = 0
    validation_by_fold = []
    for fold, (fit_pos, hold_pos) in enumerate(KFold(5, shuffle=True, random_state=42).split(train), 1):
        fit, hold = train.iloc[fit_pos], train.iloc[hold_pos]
        pre = pipeline.make_preprocessor(engineered=True, scale=True).set_output(transform="pandas")
        model = scoring.MODELS["LocalOutlierFactor"](**params)
        model.fit(pre.fit_transform(fit).to_numpy())
        oof.loc[hold.index, "fold"] = fold
        oof.loc[hold.index, "lof"] = scoring.anomaly_scores(model, pre.transform(hold))
        oof.loc[hold.index, "h"] = joint_low(fit, hold)
        validation_by_fold.append({
            "h": joint_low(fit, validation),
            "lof": scoring.anomaly_scores(model, pre.transform(validation)),
        })
    if not np.isfinite(oof.to_numpy()).all() or oof.fold.value_counts().to_dict() != dict.fromkeys(range(1, 6), 1400):
        raise ValueError("Incomplete out-of-fold reference.")
    stability = []
    for fold, fold_scores in enumerate(validation_by_fold, 1):
        for name, values in fold_scores.items():
            full = full_scores[name]
            drift = percentile(oof[name], full) - percentile(oof[name], values)
            tail = full >= full.quantile(0.95)
            stability.append(dict(fold=fold, channel=name, spearman=full.corr(values, method="spearman"),
                                  fold_q95=values.quantile(.95), full_q95=full.quantile(.95),
                                  median_rank_shift=drift.median(), p95_abs_rank_shift=drift.abs().quantile(.95),
                                  top5pct_median_rank_shift=drift[tail].median()))
    return oof, pd.DataFrame(stability)


def run():
    frozen_dir, manifest = freezing.read_manifest(ROOT / "results/frozen", BASELINE)
    prior, heat_manifest = freezing.read_manifest(ROOT / "results/heat_priority1", HEAT_RUN)
    baseline = manifest["baseline"]
    if (heat_manifest["baseline"] != BASELINE or pipeline.fingerprint() != baseline["data_sha256"]
            or pipeline.split_fingerprint() != baseline["indices_sha256"]
            or scoring.file_digest(ROOT / "config.json") != manifest["config_sha256"]):
        raise ValueError("Baseline data or protocol mismatch.")
    train, _, _ = pipeline.load_split("train", engineered=True)
    val, y, modes = pipeline.load_split("validation", engineered=True)
    if (len(train), len(val)) != (7000, 1500) or not train.index.is_unique or not train.index.intersection(val.index).empty:
        raise ValueError("Unexpected development split.")
    prior_scores = pd.read_csv(prior / "predictions.csv", index_col="row_index", float_precision="round_trip")
    record = manifest["selected"][LOF_ID]
    fitted = joblib.load(frozen_dir / "models.joblib")[LOF_ID]
    scores = pd.DataFrame({"h": joint_low(train, val), "lof": freezing.frozen_scores(val, record, fitted)})
    if (not scores.h.equals(prior_scores.h_only_score) or not scores.lof.equals(prior_scores.baseline_LocalOutlierFactor_score)
            or not y.equals(prior_scores.y_true) or not modes.index.equals(y.index)):
        raise ValueError("Single-channel validation replay differs from priority 1.")
    oof, stability = build_reference(train, val, scores, record["params"])
    for channel in ("h", "lof"):
        scores[channel + "_percentile"] = percentile(oof[channel], scores[channel])
    scores["fusion"] = np.maximum(scores.h_percentile, scores.lof_percentile)
    hdf, pwf = y.eq(1) & modes.HDF.eq(1), y.eq(1) & modes.PWF.eq(1)
    predictions = pd.concat([y.rename("y_true"), modes[["HDF", "PWF"]], scores], axis=1)
    summary, budgets, changes = [], [], []
    for name in ("h", "lof", "fusion"):
        values = scores[name]
        check_alignment(val, y, values)
        threshold = freezing.best_f1_threshold(y, values)
        alarm = scoring.alarms_at_threshold(values, threshold)
        if name == "lof" and threshold != record["threshold"]:
            raise ValueError("LOF baseline threshold mismatch.")
        predictions[name + "_alarm"] = alarm.astype(int)
        rows = budget_rows(name, values, y, hdf, pwf)
        budgets.extend(rows)
        summary.append(dict(channel=name, threshold=threshold, **scoring.ranking_metrics(y, values),
                            **freezing.alarm_metrics(y, alarm), hdf=int((alarm & hdf).sum()),
                            pwf=int((alarm & pwf).sum())))
    for comparator in ("h", "lof"):
        for point in ("f1", "top41_including_ties"):
            if point == "f1":
                base = predictions[comparator + "_alarm"].astype(bool)
                new = predictions.fusion_alarm.astype(bool)
            else:
                base = scores[comparator] >= scores[comparator].nlargest(41).min()
                new = scores.fusion >= scores.fusion.nlargest(41).min()
            row = dict(comparator=comparator, operating_point=point, base_alarms=int(base.sum()), fusion_alarms=int(new.sum()))
            for kind, mask in (("tp", y.eq(1)), ("fp", y.eq(0)), ("hdf", hdf), ("pwf", pwf)):
                row[kind + "_added"] = int((new & ~base & mask).sum())
                row[kind + "_lost"] = int((~new & base & mask).sum())
                row[kind + "_kept"] = int((new & base & mask).sum())
            changes.append(row)

    output = ROOT / "results/fusion_priority2"
    output.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=output) as tmp:
        staged = Path(tmp)
        predictions.to_csv(staged / "predictions.csv", index_label="row_index", lineterminator="\n")
        oof.to_csv(staged / "oof.csv", index_label="row_index", lineterminator="\n")
        for name, table in (("metrics", summary), ("budgets", budgets), ("changes", changes), ("scale", stability)):
            pd.DataFrame(table).to_csv(staged / (name + ".csv"), index=False, lineterminator="\n")
        log = dict(baseline=BASELINE, heat_run=HEAT_RUN, model=LOF_ID, params=record["params"],
                   reference="KFold(5, shuffle=True, random_state=42); train-only out-of-fold h and LOF",
                   fusion="max(ECDF_oof_h(h), ECDF_oof_lof(lof)); right-inclusive empirical CDF",
                   data_sha256=baseline["data_sha256"], indices_sha256=baseline["indices_sha256"],
                   source_sha256=scoring.file_digest(Path(__file__)),
                   artifact_sha256={p.name: scoring.file_digest(p) for p in sorted(staged.iterdir())})
        path = staged / "manifest.json"
        path.write_bytes((json.dumps(log, indent=2) + "\n").encode())
        dest = output / scoring.file_digest(path)
        dest.mkdir(exist_ok=True)
        for name in (*log["artifact_sha256"], "manifest.json"):
            (staged / name).replace(dest / name)
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(budgets).query("budget == 41").to_string(index=False))
    print(pd.DataFrame(changes).to_string(index=False))
    print(stability.to_string(index=False))
    print(f"Saved {dest}")


if __name__ == "__main__":
    run()
