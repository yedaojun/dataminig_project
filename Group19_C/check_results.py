"""Independent checks for C's frozen predictions and B/C row alignment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from Group19_A.pipeline import load_split

OUT = Path(__file__).resolve().parent / "results"


def check_partition(partition):
    predictions = pd.read_csv(OUT / f"{partition}_predictions.csv", float_precision="round_trip")
    summary = pd.read_csv(OUT / f"{partition}_metrics.csv", float_precision="round_trip")
    _, labels, _ = load_split(partition, allow_test=partition == "test")
    expected = pd.Series(labels.to_numpy(), index=labels.index)
    assert len(summary) == 12 and predictions.candidate.nunique() == 12
    for _, row in summary.iterrows():
        p = predictions[predictions.candidate == row.candidate].set_index("row_index")
        assert len(p) == len(expected) and p.index.is_unique and set(p.index) == set(expected.index)
        assert p.y_true.equals(expected.loc[p.index].rename("y_true"))
        alarm = (p.score >= row.threshold).astype(int)
        assert np.array_equal(alarm.to_numpy(), p.alarm.to_numpy())
        tp = int(((p.y_true == 1) & (alarm == 1)).sum())
        fp = int(((p.y_true == 0) & (alarm == 1)).sum())
        assert (tp, fp, 1500 - 51 - fp, 51 - tp) == (row.tp, row.fp, row.tn, row.fn)
        assert np.isclose(average_precision_score(p.y_true, p.score), row.average_precision)
        assert np.isclose(roc_auc_score(p.y_true, p.score), row.roc_auc)
    return predictions


def main():
    manifest = json.loads((OUT / "freeze_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "frozen_before_test" and len(manifest["selected"]) == 12
    check_partition("validation")
    test = check_partition("test")
    b_files = list((OUT.parent.parent / "Group19_B/results/final_test").glob("*/predictions.csv"))
    assert len(b_files) == 1
    b = pd.read_csv(b_files[0])
    b = b[b.feature_set.isin(["raw", "engineered"])]
    for _, group in b.groupby("candidate"):
        c = test[test.feature_set == group.feature_set.iloc[0]].iloc[:1500]
        assert set(group.row_index) == set(c.row_index)
        assert group.set_index("row_index").y_true.equals(c.set_index("row_index").y_true)
    print("PASS: 12 models on both partitions; metrics, labels, alarms and B/C indices align")


if __name__ == "__main__":
    main()
