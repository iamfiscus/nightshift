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

## Prior Findings (from 67 previous experiments)

A previous run discovered the following — use this as your starting point, not the defaults:

**Best config found (MAE=12,271,108):**
- context_factor: 16 (longer context is critical)
- max_rows: 10000 (use all available data)
- num_train_samples: 500 (max sampling)
- train_batch_size: 32 (larger batches help)
- learning_rate: 2.5e-5 (lower than default)
- warmup_steps: 100, stable_steps: 800, decay_steps: 100 (long stable phase)
- covariates: [airtemperature] (helps WITH enough context/data)

**Key insights:**
- Covariates only help when context_factor=16 and max_rows=10000. With short context or little data, airtemperature is noise.
- Lower learning rates (2-5e-5) outperform higher ones.
- Longer stable_steps (800) with short warmup/decay (100 each) works best.
- The model improved from MAE=21.6M (defaults) to MAE=12.3M (43% improvement).

**What to explore next:**
- Fine-tune learning_rate between 1e-5 and 5e-5
- Try stable_steps between 600-1000
- Test if even more max_steps (>1000) helps
- Explore warmup_steps between 50-200
- Try batch_size 4 or 8 with current best (32 was best but worth confirming)

## Strategy Guidelines
1. Start with the best known config above as your baseline
2. Change ONE thing at a time to isolate effects
3. If a change improves MAE, keep it and build on it
4. If a change worsens MAE, revert it
5. Focus on fine-tuning near the known optimum — don't waste runs on configurations already tested
6. Log your reasoning for each experiment
