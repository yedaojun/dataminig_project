# Group 19 predictive-maintenance project

This repository contains the shared data contract and the four workstreams for
the AI4I 2020 predictive-maintenance project.

| Workstream | Scope | Entry point |
|---|---|---|
| A | Frozen split, preprocessing, feature construction, data audit | [`Group19_A/HANDOFF.md`](Group19_A/HANDOFF.md) |
| B | Unsupervised failure detection | [`Group19_B/README.md`](Group19_B/README.md) |
| C | Supervised binary failure detection | [`Group19_C/README.md`](Group19_C/README.md) |
| D | Five-mode diagnosis and sensor-noise robustness | [`Group19_D/README.md`](Group19_D/README.md) |

For D, run from the repository root:

```shell
python -m Group19_D.run tune
python -m Group19_D.run evaluate-test
python -m Group19_D.noise
python -m Group19_D.extended_analysis
python -m Group19_D.oracle_analysis
python -m Group19_D.check_results
```

The corruption extension requires Group C's XGBoost dependency and is run in a
separate Conda environment:

```shell
conda env create -f Group19_D/environment_corruption.yml
conda run -n aiaa3111_corruption python -B -m Group19_D.corruption_analysis
conda run -n aiaa3111_corruption python -B -m Group19_D.label_conflict_sensitivity
```

The report-ready interpretation and limitations are in
[`Group19_D/D_REPORT.md`](Group19_D/D_REPORT.md). The incoming A/B/C handoff is
[`D同学交接说明.md`](D同学交接说明.md).
