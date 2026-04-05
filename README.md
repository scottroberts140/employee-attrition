# Employee Attrition Experiments

This repository contains a configuration-driven machine learning workflow for the IBM HR Employee Attrition dataset. The project trains multiple model configurations, tracks runs with MLflow, records data version information from DVC pointer files when available, and supports programmatic comparison of completed runs.

## Overview

The workflow is organized around three configuration layers:

- Dataset configuration: shared dataset path, target, feature lists, and split settings.
- Suite configuration: one or more scenarios to run from the command line.
- Model configuration: model family defaults plus named hyperparameter variants.

Current model families:

- Random Forest
- Logistic Regression
- Gradient Boosting

## Repository Layout

```text
employee-attrition/
├── configs/
│   ├── datasets/
│   │   └── employee_attrition.yaml
│   └── models/
│       └── employee_attrition/
│           ├── rf.yaml
│           ├── lr.yaml
│           └── gb.yaml
├── data/
│   ├── processed/
│   └── raw/
├── experiments/
│   └── initial.yaml
├── src/
│   ├── compare_experiments.py
│   ├── evaluation.py
│   ├── experiment.py
│   ├── monitor_drift.py
│   ├── preprocessing.py
│   └── train.py
├── MONITORING.md
├── tests/
│   ├── test_dataset.py
│   ├── test_experiment_train_evaluation.py
│   ├── test_model_validation.py
│   └── test_preprocessing.py
└── requirements.txt
```

Generated directories such as `mlruns/`, `models/`, and `metrics/` are created as you run experiments and optional local artifact saves.

## Environment Setup

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The dataset configuration for this project is defined in `configs/datasets/employee_attrition.yaml`.

Key settings include:

- Raw dataset path: `./data/raw/WA_Fn-UseC_-HR-Employee-Attrition.CSV`
- Processed dataset path: `./data/processed/WA_Fn-UseC_-HR-Employee-Attrition.CSV`
- Target column: `Attrition`
- Numeric feature list
- Categorical feature list
- Train/test split settings

If the data files are managed with DVC and are not present locally, pull them before running experiments.

```bash
dvc pull
```

This project is intended to use the DVC-tracked training dataset for both local runs and CI. The repository is configured to use DagsHub storage as the DVC remote.

## Configuration Model

### Dataset Config

The dataset config contains shared information used by every run:

- input paths
- target column
- numeric and categorical feature definitions
- features to drop globally
- split settings such as `test_size` and `random_state`

### Suite Config

The suite file in `experiments/` controls what is run from the CLI.

The current suite is `experiments/initial.yaml`, which contains a `scenarios` section. Each scenario can define:

- `title`: MLflow experiment name
- `description`: human-readable explanation
- `defaults`: settings shared across all runs in the scenario
- `models`: a list of model config files and optional named configurations to include

If `configurations: []` is used for a model entry, all named configurations in that model file are included.

### Model Configs

Each model file in `configs/models/employee_attrition/` includes:

- `_global_`: model-family defaults shared by all named configurations
- one or more named configurations such as `baseline`, `balanced_classes`, or `slower_learning`

The merged run configuration is built in this order:

1. Dataset config
2. Scenario defaults
3. Model `_global_` config
4. Named model configuration

## Running Experiments

The main entry point is `src/experiment.py`.

Run all scenarios in a suite:

```bash
python src/experiment.py --suite initial
```

Run one scenario from a suite:

```bash
python src/experiment.py --suite initial --scenario initial
```

Run the dedicated CI scenario:

```bash
python src/experiment.py --suite initial --scenario ci --fail-on-thresholds
```

What happens during a run:

- the suite, dataset, and model configs are merged
- the dataset is loaded
- selected features are dropped
- missing values and categorical encoding are handled in preprocessing
- the target is binary encoded
- the model is trained and evaluated
- parameters, metrics, data version, and the model are logged to MLflow
- optional local model and metrics files are saved if enabled in config

## MLflow Tracking

Runs are tracked in a local `mlruns/` directory by default.

Start the MLflow UI from the project root:

```bash
mlflow ui
```

Then open the UI in your browser to inspect experiments, runs, parameters, metrics, and logged models.

Each run logs:

- flattened merged configuration parameters
- evaluation metrics
- the trained model artifact
- dataset version metadata

When a `.dvc` sidecar exists for the configured dataset, the DVC `md5` value is logged as the data version. Otherwise, the data file name is logged.

## Comparing Runs Programmatically

The project supports programmatic MLflow comparison in two ways:

- `src/experiment.py --analyze` for the integrated workflow
- `src/compare_experiments.py` as a dedicated comparison entry point that calls `mlflow.search_runs()` directly

Analyze a known MLflow experiment name by metric:

```bash
python src/experiment.py --analyze --experiment-name "Initial comparision" --metric f1
```

Equivalent dedicated wrapper script:

```bash
python src/compare_experiments.py --experiment-name "Initial comparision" --metric f1
```

Analyze by suite and scenario:

```bash
python src/experiment.py --analyze --suite initial --scenario initial --metric f1
```

Or use the dedicated wrapper with the same suite and scenario resolution:

```bash
python src/compare_experiments.py --suite initial --scenario initial --metric f1
```

Limit the report to a different number of top runs:

```bash
python src/experiment.py --analyze --suite initial --scenario initial --metric accuracy --top-n 3
```

The analysis report includes:

- top runs ranked by the selected metric
- the best run overall
- average and best metric values grouped by model type

Important detail: if you analyze by `--suite` and `--scenario`, the MLflow experiment name is taken from the scenario `title` in the suite YAML. If you rename that title after runs already exist, use `--experiment-name` to query the older MLflow experiment directly.

## Drift Monitoring

The project includes a drift monitoring script at `src/monitor_drift.py`.

It:

- loads the DVC-tracked training dataset as the reference distribution
- creates a simulated production dataset from a held-out split of the same dataset
- injects synthetic drift into selected production features
- runs Evidently drift detection across all active model input features
- prints drifted features and the overall drift share
- saves an HTML report to `reports/drift_report.html`
- exits with code `1` when drift share exceeds the configured threshold

Example:

```bash
python src/monitor_drift.py --max-drift-share 0.15
```

The written monitoring analysis is in `MONITORING.md`.

## Preprocessing and Evaluation

### Preprocessing

`src/preprocessing.py` includes helpers for:

- missing-value handling
- normalization
- binary encoding
- categorical one-hot encoding
- outlier detection and removal
- dataframe validation and quality checks

The numeric missing-value handler supports these strategies:

- `median`
- `mean`
- `mean_median`
- `drop`

### Training

`src/train.py` is responsible for:

- loading data
- validating required columns
- cleaning and encoding features
- splitting into train and test sets
- training the configured model family

### Evaluation

`src/evaluation.py` computes only the metrics requested in the merged configuration. The current metric set used in configs includes:

- `accuracy`
- `precision`
- `recall`
- `f1`
- `auc_roc`

Thresholds can be configured per metric. If a threshold is present and a run falls below it, a warning is printed during evaluation.

## Testing

Run the full test suite with:

```bash
pytest
```

Run a specific test module:

```bash
pytest tests/test_preprocessing.py
pytest tests/test_dataset.py
pytest tests/test_model_validation.py
pytest tests/test_experiment_train_evaluation.py
```

The tests cover:

- preprocessing behaviors and validation
- dataset validation against the DVC-tracked training dataset
- model validation on small controlled examples
- estimator parameter filtering
- evaluation metric handling
- suite/scenario config merging
- MLflow logging orchestration
- local artifact path resolution
- run-analysis helpers

## Current Experiment Set

The active suite in `experiments/initial.yaml` defines two scenarios:

- `initial`: the broader comparison scenario that includes all named Random Forest, Logistic Regression, and Gradient Boosting configurations.
- `ci`: a stable single-run scenario used by GitHub Actions.

The `ci` scenario runs:

- the `weaker_regularization` configuration from `lr.yaml`
- against the DVC-tracked training dataset defined in `configs/datasets/employee_attrition.yaml`
- with local artifact output enabled for `models/model.pkl` and `metrics/results.json`

The `initial` scenario includes:

- all named Random Forest configurations from `rf.yaml`
- all named Logistic Regression configurations from `lr.yaml`
- all named Gradient Boosting configurations from `gb.yaml`

Because each model entry uses `configurations: []`, running that scenario launches every named configuration in those three model files.

## DVC Remote Notes

The repository expects the training dataset to be tracked with DVC and restored with `dvc pull`.

This repository uses DagsHub storage as the DVC remote.

Remote configuration summary:

- DVC remote name: `origin`
- DagsHub repository: `https://dagshub.com/scottrdeveloper/employee-attrition`
- The workflow authenticates to the DVC remote with the `DAGSHUB_TOKEN` GitHub secret

The workflow uses the `DAGSHUB_TOKEN` secret to configure DVC authentication on the runner before calling `dvc pull`.

To configure DVC authentication locally:

```bash
./.venv/bin/dvc remote modify --local origin access_key_id <your-dagshub-token>
./.venv/bin/dvc remote modify --local origin secret_access_key <your-dagshub-token>
```

Then verify data sync with:

```bash
./.venv/bin/dvc pull
./.venv/bin/dvc status -c
```

## Notes

- The project currently uses MLflow's local filesystem backend for tracking.
- The raw dataset in this repo includes intentionally injected missing values for preprocessing and testing work.
- Local artifact saving is optional and controlled through configuration with `save_local_artifacts` and `local_output_paths`.
