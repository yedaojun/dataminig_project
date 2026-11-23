"""Recalculate key invariants of C's extended evidence tables."""
import numpy as np
import pandas as pd
from Group19_C.extended_analysis import OUT, chosen_candidates, read_predictions
from sklearn.metrics import average_precision_score


def main():
    choices = chosen_candidates()
    validation = read_predictions("validation", choices)
    test = read_predictions("test", choices)
    boot = pd.read_csv(OUT / "paired_bootstrap.csv")
    assert len(boot) == 7 and (boot.bootstrap_repeats == 2000).all()
    for name, frame in test.items():
        row = boot[boot.contrast == name].iloc[0]
        assert np.isclose(row.point, average_precision_score(frame.y_true, frame.score))
        assert np.isfinite(row.ci_low) and np.isfinite(row.ci_high)
    budgets = pd.read_csv(OUT / "alarm_budgets.csv", float_precision="round_trip")
    assert len(budgets) == 18
    for _, row in budgets.iterrows():
        frame = (validation if row.partition == "validation" else test)[row.model]
        alarm = frame.score.to_numpy() >= row.threshold
        y = frame.y_true.to_numpy()
        tp = int(((y == 1) & alarm).sum())
        fp = int(((y == 0) & alarm).sum())
        assert (tp, fp) == (row.tp, row.fp)
        if row.partition == "validation":
            assert fp <= row.allowed_validation_fp
    detail = pd.read_csv(OUT / "label_efficiency_repeats.csv")
    summary = pd.read_csv(OUT / "label_efficiency_summary.csv")
    assert len(detail) == 32 and len(summary) == 8
    original = pd.read_csv(OUT / "validation_metrics.csv")
    for _, row in summary[summary.train_fraction == 1].iterrows():
        candidate = choices[row.model][1]
        reference = original.loc[original.candidate == candidate, "average_precision"].iloc[0]
        assert np.isclose(row.ap_mean, reference, atol=1e-12)
    assert (detail.train_positives > 0).all()
    print("PASS: bootstrap points, alarm counts/budgets and label-efficiency baselines verified")


if __name__ == "__main__":
    main()
