"""Paired validation sensitivity analysis of saved priority-2 scores, without refitting."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from Group19_B import freezing, scoring
from Group19_B.heat_experiment import ROOT
from Group19_B.wear_experiment import FUSION_RUN


CHANNELS = ("fusion", "h", "lof")
METRICS = ("ap", "fixed_f1", "tp_at41", "tp_at41_low", "tp_at41_high")
BUDGET = 41
REPEATS = 5000
SEED = 42


class RankedScores:
    """Keep the original ordering and score ties while varying each row's multiplicity."""

    def __init__(self, scores, labels, threshold):
        self.order = np.argsort(-scores, kind="stable")
        sorted_scores = scores[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
        self.y = labels[self.order]
        self.alarm = sorted_scores >= threshold

    def evaluate(self, multiplicities):
        weights = multiplicities[self.order]
        positive = weights * self.y
        npos = positive.sum()
        hits = np.add.reduceat(positive, self.starts)
        cum_hits = hits.cumsum()
        cum_n = np.add.reduceat(weights, self.starts).cumsum()
        precision = np.divide(cum_hits, cum_n, out=np.zeros(len(hits)), where=cum_n > 0)
        ap = (hits @ precision) / npos
        f1 = 2 * positive[self.alarm].sum() / (npos + weights[self.alarm].sum())

        boundary = np.searchsorted(cum_n, BUDGET)
        before_n = cum_n[boundary - 1] if boundary else 0
        before_tp = cum_hits[boundary - 1] if boundary else 0
        tied_n = cum_n[boundary] - before_n
        slots = BUDGET - before_n
        tied_tp = hits[boundary]
        expected = before_tp + slots * tied_tp / tied_n  # uniform tie-breaking, diagnostic only
        low = before_tp + max(0, slots - (tied_n - tied_tp))
        high = before_tp + min(slots, tied_tp)
        return ap, f1, expected, low, high


def run():
    source, _ = freezing.read_manifest(ROOT / "results/fusion_priority2", FUSION_RUN)
    p = pd.read_csv(source / "predictions.csv", float_precision="round_trip").set_index("row_index", verify_integrity=True)
    metrics = pd.read_csv(source / "metrics.csv", float_precision="round_trip").set_index("channel")
    if len(p) != 1500 or p.y_true.value_counts().to_dict() != {1: 51, 0: 1449}:
        raise ValueError("Unexpected validation sample.")
    y = p.y_true.to_numpy()
    channels = {}
    for name in CHANNELS:
        values = p[name].to_numpy()
        threshold = metrics.at[name, "threshold"]
        if not np.isfinite(values).all() or not np.array_equal(values >= threshold, p[name + "_alarm"]):
            raise ValueError(f"Invalid saved scores or threshold for {name}.")
        channels[name] = RankedScores(values, y, threshold)

    def measure(weights):
        result = {}
        for name, channel in channels.items():
            for key, value in zip(METRICS, channel.evaluate(weights)):
                result[f"{name}_{key}"] = value
        for other in ("h", "lof"):
            for key in ("ap", "fixed_f1", "tp_at41"):
                result[f"vs_{other}_{key}"] = result[f"fusion_{key}"] - result[f"{other}_{key}"]
            # Conservative difference bounds if a tie straddles the budget for either channel.
            result[f"vs_{other}_tp_at41_low"] = result["fusion_tp_at41_low"] - result[f"{other}_tp_at41_high"]
            result[f"vs_{other}_tp_at41_high"] = result["fusion_tp_at41_high"] - result[f"{other}_tp_at41_low"]
        return result

    weights = np.ones(len(p), dtype=int)
    observed = measure(weights)
    for name in CHANNELS:
        if not np.allclose([observed[f"{name}_ap"], observed[f"{name}_fixed_f1"]],
                           metrics.loc[name, ["average_precision", "f1"]].to_numpy(), atol=1e-14, rtol=0):
            raise ValueError("Original validation metrics could not be reproduced.")
    rng = np.random.default_rng(SEED)
    positives, negatives = np.flatnonzero(y), np.flatnonzero(y == 0)
    samples = []
    for _ in range(REPEATS):
        rows = np.r_[rng.choice(positives, len(positives)), rng.choice(negatives, len(negatives))]
        samples.append(measure(np.bincount(rows, minlength=len(p))))
    bootstrap = pd.DataFrame(samples)

    deletions = []
    for i, row_id in enumerate(p.index):
        weights[i] = 0
        deletions.append(dict(unit="row", removed_id=row_id, removed_n=1, removed_faults=y[i], **measure(weights)))
        weights[i] = 1
    # Sections of the original row order, not chronological windows of known duration.
    for block in range(10):
        removed = (p.index.to_numpy() // 1000) == block
        deletions.append(dict(unit="block", removed_id=block, removed_n=removed.sum(),
                              removed_faults=y[removed].sum(), **measure((~removed).astype(int))))
    deletion = pd.DataFrame(deletions)

    summaries = []
    for key, obs in observed.items():
        values = bootstrap[key]
        lower, median, upper = values.quantile([.025, .5, .975])
        row = dict(measure=key, observed=obs, bootstrap_p025=lower, bootstrap_median=median,
                   bootstrap_p975=upper, bootstrap_share_gt_zero=(values > 0).mean())
        for unit in ("row", "block"):
            v = deletion.loc[deletion.unit.eq(unit), key]
            row.update({f"{unit}_min": v.min(), f"{unit}_max": v.max(),
                        f"{unit}_n_gt_zero": int((v > 0).sum())})
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    output = ROOT / "results/stability_priority6"
    output.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=output) as tmp:
        stage = Path(tmp)
        summary.to_csv(stage / "summary.csv", index=False, lineterminator="\n")
        bootstrap.to_csv(stage / "bootstrap.csv", index_label="replicate", lineterminator="\n")
        deletion.to_csv(stage / "deletions.csv", index=False, lineterminator="\n")
        manifest = dict(candidate_run=FUSION_RUN, validation_rows=len(p), positives=int(y.sum()),
                        thresholds=metrics.threshold.to_dict(), seed=SEED, paired_stratified_repeats=REPEATS,
                        delete_blocks="zero-based original row index // 1000, blocks 0 through 9",
                        budget=BUDGET, budget_ties="expected hits for uniform ties; lower and upper counts also recorded",
                        source_sha256=scoring.file_digest(Path(__file__)),
                        artifact_sha256={a.name: scoring.file_digest(a) for a in sorted(stage.iterdir())})
        path = stage / "manifest.json"
        path.write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
        dest = output / scoring.file_digest(path)
        dest.mkdir(exist_ok=True)
        for name in (*manifest["artifact_sha256"], "manifest.json"):
            (stage / name).replace(dest / name)
    print(summary[summary.measure.str.startswith(("fusion_", "vs_"))].to_string(index=False))
    print(deletion.loc[deletion.unit.eq("block"), ["removed_id", "removed_n", "removed_faults", "vs_h_ap",
                                                   "vs_h_fixed_f1", "vs_h_tp_at41"]].to_string(index=False))
    print(f"Saved {dest}")


if __name__ == "__main__":
    run()
