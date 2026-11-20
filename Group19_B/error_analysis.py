"""Reconstruct stage-six error counts and cases from the accepted test export."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from Group19_A import pipeline
from Group19_B import freezing, scoring


ROOT = Path(__file__).resolve().parent
TEST_RUN = "f794ac36dc8d6e14cba353c6f79d10cfa74094617c07e942814de540852b043e"
PAIRS = [
    ("raw", "raw_IsolationForest_6", "raw_LocalOutlierFactor_7"),
    ("engineered", "engineered_IsolationForest_5", "engineered_LocalOutlierFactor_7"),
    ("fusion_vs_engineered_lof", "heat_lof_percentile_max", "engineered_LocalOutlierFactor_7"),
]


def run():
    source, log = freezing.read_manifest(ROOT / "results/final_test", TEST_RUN)
    if (log["data_sha256"] != pipeline.fingerprint() or log["indices_sha256"] != pipeline.split_fingerprint()):
        raise ValueError("Case data or row IDs differ from the accepted test run.")
    x, y, modes = pipeline.load_split("test", allow_test=True)
    p = pd.read_csv(source / "predictions.csv", float_precision="round_trip")
    metrics = pd.read_csv(source / "metrics.csv", float_precision="round_trip").set_index("candidate")
    blocks = {}
    for cid, frame in p.groupby("candidate", sort=False):
        g = frame.set_index("row_index", verify_integrity=True)
        if (not y.equals(g.y_true) or not g.partition.eq("test").all()
                or not g.threshold.eq(log["thresholds"][cid]).all()
                or not np.array_equal(g.alarm, g.score.ge(g.threshold))):
            raise ValueError(f"Test row, label or working point mismatch for {cid}.")
        result = dict(**freezing.alarm_metrics(y, g.alarm), **scoring.ranking_metrics(y, g.score))
        if result != metrics.loc[cid, list(result)].to_dict():
            raise ValueError(f"Test summary mismatch for {cid}.")
        blocks[cid] = g

    pairs, cases, by_mode = [], [], []
    for group, first, second in PAIRS:
        a, b = blocks[first].alarm.eq(1), blocks[second].alarm.eq(1)
        for label in (0, 1):
            subset = y.eq(label)
            pairs.append(dict(comparison=group, first=first, second=second, y_true=label,
                              both=int((subset & a & b).sum()), first_only=int((subset & a & ~b).sum()),
                              second_only=int((subset & ~a & b).sum()), neither=int((subset & ~a & ~b).sum())))
        for focal, peer in ((first, second), (second, first)):
            g, other = blocks[focal], blocks[peer]
            rules = [
                ("highest_score_false_positive", y.eq(0) & g.alarm.eq(1), False),
                ("lowest_score_false_negative", y.eq(1) & g.alarm.eq(0), True),
                ("highest_score_exclusive_hit", y.eq(1) & g.alarm.eq(1) & other.alarm.eq(0), False),
            ]
            for rule, mask, ascending in rules:
                eligible = g.loc[mask].sort_index()
                if eligible.empty:
                    continue
                idx = eligible.score.sort_values(ascending=ascending, kind="stable").index[0]
                cases.append(dict(comparison=group, candidate=focal, comparator=peer, selection=rule,
                                  row_index=idx, **x.loc[idx].to_dict(), y_true=y.at[idx],
                                  **modes.loc[idx].to_dict(), score=g.at[idx, "score"],
                                  threshold=g.at[idx, "threshold"], alarm=g.at[idx, "alarm"],
                                  comparator_score=other.at[idx, "score"], comparator_threshold=other.at[idx, "threshold"],
                                  comparator_alarm=other.at[idx, "alarm"]))
    for mode, flag in {**dict(modes.items()), "no_mode_flag": modes.sum(axis=1).eq(0)}.items():
        subset = y.eq(1) & flag.eq(1)
        for cid, g in blocks.items():
            by_mode.append(dict(mode=mode, candidate=cid, positive_support=int(subset.sum()),
                                detected=int(g.alarm[subset].sum()), missed=int(g.alarm[subset].eq(0).sum())))
    output = ROOT / "results/error_analysis" / TEST_RUN
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temp:
        for name, rows in (("paired_counts", pairs), ("cases", cases), ("positive_modes", by_mode)):
            path = Path(temp) / (name + ".csv")
            pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")
            path.replace(output / path.name)
    print(f"PASS: verified five test summaries; exported {len(cases)} case selections to {output}")


if __name__ == "__main__":
    run()
