# Drift Monitoring

The project includes a drift monitoring script at `src/monitor_drift.py`.

## How it works

- loads the DVC-tracked training dataset defined in `configs/datasets/employee_attrition.yaml`
- uses the training split as the reference distribution
- creates a simulated production batch from the held-out portion of the dataset
- injects synthetic drift into selected features
- runs Evidently drift detection on all active model input features
- prints the drifted features and overall drift share
- saves an HTML report to `reports/drift_report.html`
- exits with code `1` when drift share exceeds the configured threshold

Example:

```bash
python src/monitor_drift.py --max-drift-share 0.15
```

## Analysis

### Which features showed drift and why?

The monitoring script deliberately introduces drift into features that are plausible drivers of employee attrition behavior and likely to change after deployment:

- `MonthlyIncome` is shifted upward to simulate a compensation mix change in the incoming population.
- `DistanceFromHome` is shifted upward to simulate a different geographic hiring profile.
- `Age` is shifted for part of the production batch to simulate a change in workforce demographics.
- `OverTime` is shifted toward `Yes` to simulate a higher-workload environment.
- `Department` is shifted to simulate a change in hiring distribution across departments.

These are the features most likely to appear as drifted because the script modifies their distributions directly before running Evidently.

### Would this drift likely affect model performance?

Yes. Several of the simulated drifted features are strong behavioral or demographic signals for attrition, especially `MonthlyIncome`, `OverTime`, `DistanceFromHome`, and `Age`. If their production distributions differ materially from the training data, the model may make predictions in regions of feature space it saw less frequently during training. That can reduce precision, recall, or calibration even if aggregate accuracy appears stable for a period of time.

### What action would you recommend?

The recommended action is to investigate first, then retrain if the drift persists or is accompanied by degraded business metrics. A reasonable sequence is:

- confirm the drift reflects a real operational change rather than a data pipeline issue
- review prediction quality on recent labeled outcomes if available
- continue monitoring if the drift is modest and model quality remains stable
- retrain the model if drift remains elevated or performance metrics begin to decline
