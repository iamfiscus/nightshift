# Nightshift

An autonomous AI research agent that optimizes time series forecasting overnight. A local SLM (via Ollama) acts as the scientist — proposing experiment configurations, analyzing results, and iterating — while TOTO fine-tuning runs on a remote GPU.

**The pitch:** You go to sleep. Nightshift runs 100 experiments. You wake up to a leaderboard showing what the agent discovered.

## Architecture

```
YOUR MACHINE (Mac / Gertie)              RUNPOD (GPU)
┌──────────────────────────┐             ┌──────────────────────┐
│  run.py (CLI)            │             │  train_remote.py     │
│    │                     │    SSH      │    │                 │
│    ├─► Scientist (Ollama)│────────────►│    ├─► TOTO fine-tune│
│    │   reads results     │◄────────────│    ├─► Evaluate      │
│    │   proposes config   │  metrics.json    └─► metrics.json  │
│    │                     │             │                      │
│    ├─► Logger            │             │  Dataset: proenfo_gfc12
│    │   experiments/      │             │  (auto-downloaded from HF)
│    │   leaderboard.md    │             └──────────────────────┘
│    │                     │
│    └─► Executor (SSH)    │
└──────────────────────────┘
```

## Quick Start

### 1. Install locally (on Gertie or your Mac)

```bash
git clone https://github.com/jdfiscus/nightshift.git
cd nightshift
pip install pyyaml rich paramiko ollama tabulate
```

### 2. Pull the scientist model

```bash
# On Gertie (128GB RAM):
ollama pull granite4:small-h

# On Mac (less RAM):
ollama pull qwen3.5:9b
```

### 3. Set up RunPod

1. Create a RunPod instance: **RTX 4090** ($0.39/hr), **PyTorch 2.x** template, **20GB** volume
2. Note the SSH host, port, and password from the RunPod dashboard
3. Run one-time setup:

```bash
# Upload setup script and run it
scp scripts/setup_runpod.sh root@<HOST>:/workspace/
ssh -p <PORT> root@<HOST> "bash /workspace/setup_runpod.sh"

# Upload training scripts
python run.py --host <HOST> --port <PORT> --password <PASSWORD> --setup
```

### 4. Test with 3 experiments

```bash
python run.py \
  --host <HOST> \
  --port <PORT> \
  --password <PASSWORD> \
  --model granite4:small-h \
  --num 3
```

### 5. Run overnight

```bash
nohup python run.py \
  --host <HOST> \
  --port <PORT> \
  --password <PASSWORD> \
  --model granite4:small-h \
  --num 100 \
  > nightshift.log 2>&1 &
```

### 6. Wake up and check results

```bash
cat leaderboard.md
```

## How It Works

1. **First run:** Uses default config from `experiment.yaml` as baseline
2. **Subsequent runs:** The SLM scientist reads `program.md` (research goals) + past experiment results, then proposes a modified config
3. **Each experiment:** Config is uploaded to RunPod via SSH, TOTO fine-tunes on the energy dataset, metrics are downloaded
4. **Logging:** Every experiment is saved to `experiments/exp_XXX/` with config, metrics, and the scientist's reasoning
5. **Leaderboard:** `leaderboard.md` is regenerated after each run, ranked by MAE

## Project Structure

```
nightshift/
├── run.py                  # CLI entrypoint
├── program.md              # Research goals (read by the SLM scientist)
├── experiment.yaml         # Current experiment config (modified by scientist)
├── leaderboard.md          # Generated after each run
├── agent/
│   ├── scientist.py        # Ollama SLM proposes experiment configs
│   ├── executor.py         # SSH to RunPod, run training, download metrics
│   ├── logger.py           # Log experiments, generate leaderboard
│   └── loop.py             # Orchestration: scientist → executor → logger
├── scripts/
│   ├── train_remote.py     # Runs ON RunPod: TOTO fine-tune + eval
│   ├── finetune_config.yaml# TOTO base config
│   └── setup_runpod.sh     # One-time RunPod setup
├── experiments/            # Generated: one dir per experiment
│   └── exp_001/
│       ├── config.yaml
│       ├── metrics.json
│       └── reasoning.md
└── tests/                  # 14 tests
```

## Dataset

**proenfo_gfc12** — hourly electricity load from the [FEV Benchmark](https://huggingface.co/datasets/autogluon/fev_datasets). Includes air temperature as an exogenous covariate. Auto-downloaded by TOTO on RunPod at training time.

## Cost

- **RunPod RTX 4090:** ~$0.39/hr
- **100 experiments overnight (~8 hrs):** ~$3.12
- **Ollama (local):** Free

## Inspired By

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — autonomous AI research agent pattern
- [Datadog TOTO](https://github.com/DataDog/toto) — time series foundation model with exogenous covariates
