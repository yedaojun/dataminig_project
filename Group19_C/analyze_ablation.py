"""Matched-configuration validation contrasts for feature and weight ablations."""
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent / "results"


def main():
    candidates = pd.read_csv(OUT / "candidates.csv")
    candidates["ordinal"] = candidates.candidate.str.extract(r"_(\d+)$").astype(int)
    matched = candidates.pivot(index=["model", "weighting", "ordinal", "params"],
                               columns="feature_set", values="validation_ap").reset_index()
    matched["engineered_minus_raw_ap"] = matched.engineered - matched.raw
    matched.to_csv(OUT / "matched_feature_ablation.csv", index=False)
    weights = candidates.pivot(index=["model", "feature_set", "ordinal", "params"],
                               columns="weighting", values="validation_ap").reset_index()
    weights["weighted_minus_unweighted_ap"] = weights.weighted - weights.unweighted
    weights.to_csv(OUT / "matched_weight_ablation.csv", index=False)
    assert len(matched) == len(weights) == 18
    print("PASS: 18 matched feature contrasts and 18 matched weighting contrasts")


if __name__ == "__main__":
    main()
