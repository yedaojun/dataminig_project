"""Command-line entry point for workstream B. Run from the repository root."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from Group19_A import pipeline
from Group19_B import freezing, scoring


ROOT = Path(__file__).resolve().parent
DEVELOPMENT_SPLITS = ("train", "validation")


def check_alignment(features, labels, scores=None):
    """Require score and label series to carry the feature frame's original row IDs."""
    if not features.index.is_unique or not features.index.equals(labels.index):
        raise ValueError("Feature and label row indices must be unique and in identical order.")
    if scores is not None:
        if not isinstance(scores, pd.Series) or not scores.index.equals(features.index):
            raise ValueError("Scores must carry the same ordered row indices as the features.")
        if not np.isfinite(scores.to_numpy()).all():
            raise ValueError("Scores contain non-finite values.")


def load_development_data(config, feature_set, indices):
    settings = config["feature_sets"][feature_set]
    columns = config["feature_sets"]["raw"]["columns"] + settings.get("additional_columns", [])
    splits = {}
    for name in DEVELOPMENT_SPLITS:
        features, labels, _ = pipeline.load_split(name, engineered=settings["engineered"])
        check_alignment(features, labels)
        if len(labels) != config["baseline"]["split_sizes"][name]:
            raise ValueError(f"Unexpected number of rows in {name}.")
        if not features.index.equals(indices[name]):
            raise ValueError(f"{name} row order differs from the frozen A indices.")
        if labels.name != config["target"]["column"] or not labels.isin([0, 1]).all():
            raise ValueError(f"Invalid binary target in {name}.")
        if list(features.columns) != columns or set(columns).intersection(config["target"]["excluded_inputs"]):
            raise ValueError(f"Unexpected input columns for {feature_set} in {name}.")
        splits[name] = (features, labels)
    return splits


def prepare_features(splits, engineered, scale):
    preprocessor = pipeline.make_preprocessor(engineered=engineered, scale=scale)
    preprocessor.set_output(transform="pandas")
    matrices = {
        "train": preprocessor.fit_transform(splits["train"][0]),
        "validation": preprocessor.transform(splits["validation"][0]),
    }
    expected_width = 11 if engineered else 8
    for name, matrix in matrices.items():
        check_alignment(matrix, splits[name][1])
        if matrix.shape[1] != expected_width or not matrix.columns.equals(matrices["train"].columns):
            raise ValueError(f"Unexpected transformed columns in {name}.")
        if not np.isfinite(matrix.to_numpy()).all():
            raise ValueError(f"Non-finite transformed values in {name}.")
    return preprocessor, matrices


def validate():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    baseline = config["baseline"]
    manifest_path = ROOT.parent / baseline["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("data_sha256", "indices_sha256", "seed", "stratify"):
        if manifest[key] != baseline[key]:
            raise ValueError(f"A's manifest differs from B's accepted protocol for {key}.")
    # load_split also verifies both files' actual hashes against this manifest.
    with np.load(manifest_path.with_name("indices.npz")) as archive:
        indices = {name: pd.Index(archive[name]) for name in DEVELOPMENT_SPLITS}
    if not indices["train"].intersection(indices["validation"]).empty:
        raise ValueError("Training and validation row indices overlap.")

    candidates = {}
    fitted = {"models": {}, "preprocessors": {}}
    validation_scores = pd.DataFrame(index=indices["validation"])
    for feature_set, settings in config["feature_sets"].items():
        splits = load_development_data(config, feature_set, indices)
        labels = splits["validation"][1]
        validation_scores[labels.name] = labels
        for model, model_settings in config["models"].items():
            scale = model_settings["scale_numeric"]
            preprocessor, matrices = prepare_features(splits, settings["engineered"], scale)
            group = f"{feature_set}_{model}"
            fitted["preprocessors"][group] = preprocessor
            for order, params in enumerate(model_settings["candidates"], start=1):
                candidate = f"{group}_{order}"
                estimator = scoring.MODELS[model](**model_settings["fixed_params"], **params)
                estimator.fit(matrices["train"].to_numpy())
                scores = scoring.anomaly_scores(estimator, matrices["validation"])
                check_alignment(matrices["validation"], labels, scores)
                validation_scores[candidate] = scores
                fitted["models"][candidate] = estimator
                candidates[candidate] = {
                    "model": model, "feature_set": feature_set, "order": order,
                    "preprocessor_key": group, "scale_numeric": scale,
                    "feature_names": list(matrices["train"].columns),
                    "params": estimator.get_params(),
                    "validation_metrics": scoring.ranking_metrics(labels, scores),
                }
            print(f"{feature_set} / {model}: train {matrices['train'].shape}, "
                  f"validation {matrices['validation'].shape}, scored {len(model_settings['candidates'])} candidates")

    rows = pd.concat([
        pd.DataFrame({"split": name, "row_index": index}) for name, index in indices.items()
    ], ignore_index=True)
    output = ROOT / "results" / "development_indices.csv"
    output.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as staging_dir:
        run_dir = scoring.save_validation_run(
            ROOT, config, validation_scores, candidates, fitted, Path(staging_dir) / "run"
        )
        staged = Path(staging_dir) / output.name
        rows.to_csv(staged, index=False, encoding="utf-8", lineterminator="\n")
        staged.replace(output)
    print(f"PASS: saved {len(rows)} ordered row indices to {output.relative_to(ROOT.parent)}")
    print(f"PASS: {len(candidates)} validation candidates saved to {run_dir.relative_to(ROOT.parent)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="check development inputs and score every validation candidate")
    commands.add_parser("freeze", help="calibrate validation thresholds and freeze selected fitted models")
    commands.add_parser("evaluate-test", help="evaluate the frozen selections on the held-out test split")
    command = parser.parse_args().command
    try:
        if command == "freeze":
            return freezing.freeze(ROOT)
        if command == "evaluate-test":
            from Group19_B.final_evaluation import evaluate_test
            return evaluate_test(ROOT)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    validate()


if __name__ == "__main__":
    main()
