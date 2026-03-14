# Nightshift Research Program: Energy Demand Forecasting

## Objective
Minimize **MAE** (Mean Absolute Error) on the `proenfo_gfc12` electricity load dataset
by fine-tuning TOTO with different configurations.

## Dataset
- **Name:** proenfo_gfc12 (from autogluon/fev_datasets on HuggingFace)
- **Target:** Hourly electricity load
- **Available covariates:** airtemperature
- **Forecast horizon:** 168 hours (7 days)
- **Seasonality:** 24 hours

## What You Can Modify (in experiment.yaml)
- `covariates.include` — which exogenous features to use (e.g., airtemperature)
- `training.learning_rate` — range: 1e-5 to 1e-3
- `training.warmup_steps` — range: 100 to 2000
- `training.stable_steps` — range: 100 to 1000
- `training.decay_steps` — range: 100 to 1000
- `training.max_steps` — must equal warmup + stable + decay
- `training.batch_size` — options: 4, 8, 16, 32
- `data.context_factor` — options: 4, 8, 16 (context_length = 64 * context_factor)
- `data.num_train_samples` — range: 50 to 500
- `data.max_rows` — range: 1000 to 10000

## What You Must NOT Modify
- The dataset (always proenfo_gfc12)
- The evaluation metric (MAE)
- The model (always Datadog/Toto-Open-Base-1.0)
- The forecast horizon (168)

## Strategy Guidelines
1. Start by establishing a baseline with default settings
2. Change ONE thing at a time to isolate effects
3. If a change improves MAE, keep it and build on it
4. If a change worsens MAE, revert it
5. After exploring individual parameters, try combining the best settings
6. Log your reasoning for each experiment
