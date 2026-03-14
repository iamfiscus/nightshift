# Nightshift

An autonomous AI research agent that optimizes time series forecasting overnight. An SLM (via Ollama) acts as the scientist — proposing experiment configurations, analyzing results, and iterating — while TOTO fine-tuning runs on a GPU. Everything runs on a single [RunPod](https://runpod.io?ref=de53e8ds) instance.

**The pitch:** You go to sleep. Nightshift runs 100 experiments. You wake up to a leaderboard showing what the agent discovered.

## Architecture

```
RUNPOD INSTANCE (GPU + Ollama)
┌─────────────────────────────────────────┐
│  run_local.py (CLI)                     │
│    │                                    │
│    ├─► Scientist (Ollama)               │
│    │   granite4:small-h / qwen3.5       │
│    │   reads results, proposes config   │
│    │                                    │
│    ├─► Local Executor                   │
│    │   runs train_remote.py via subprocess
│    │   TOTO fine-tune + eval on GPU     │
│    │                                    │
│    └─► Logger                           │
│        experiments/exp_XXX/             │
│        leaderboard.md                   │
│                                         │
│  Dataset: proenfo_gfc12                 │
│  (auto-downloaded from HuggingFace)     │
└─────────────────────────────────────────┘
```

## Quick Start (RunPod — Recommended)

Everything runs on one [RunPod](https://runpod.io?ref=de53e8ds) instance — no local machine needed.

### 1. Create a RunPod pod

Sign up at [RunPod](https://runpod.io?ref=de53e8ds) and launch a GPU pod:

- **GPU:** RTX 4000 Ada ($0.26/hr) or RTX 4090 ($0.59/hr)
- **Template:** RunPod PyTorch 2.4.0
- **Volume Disk:** 50GB
- **SSH terminal access:** enabled

### 2. SSH in and run setup

```bash
ssh root@<HOST> -p <PORT> -i ~/.ssh/id_ed25519
```

Then run the one-liner setup:

```bash
cd /workspace && \
git clone https://github.com/DataDog/toto.git && \
cd toto && pip install -r requirements.txt && pip install -e . && \
curl -fsSL https://ollama.com/install.sh | sh && \
cd /workspace && \
git clone https://github.com/iamfiscus/nightshift.git && \
cd nightshift && pip install pyyaml rich paramiko ollama tabulate
```

### 3. Start Ollama and pull the scientist model

```bash
ollama serve &
sleep 5
ollama pull granite4:small-h
```

### 4. Test with 3 experiments

```bash
cd /workspace/nightshift
python run_local.py --model granite4:small-h --num 3
```

### 5. Run overnight

```bash
nohup python run_local.py --model granite4:small-h --num 100 > nightshift.log 2>&1 &
```

### 6. Wake up and check results

```bash
cat leaderboard.md
```

## Split Mode (Optional)

If you prefer to run the scientist on a local machine (Mac/Gertie) and only use [RunPod](https://runpod.io?ref=de53e8ds) for GPU training:

```bash
# Local machine
git clone https://github.com/iamfiscus/nightshift.git
cd nightshift
pip install pyyaml rich paramiko ollama tabulate
ollama pull qwen3.5:9b

# Set up RunPod for remote training
python run.py --host <HOST> --port <PORT> --password <PASSWORD> --setup

# Run
python run.py --host <HOST> --port <PORT> --password <PASSWORD> --model qwen3.5:9b --num 100
```

## How It Works

1. **First run:** Uses default config from `experiment.yaml` as baseline
2. **Subsequent runs:** The SLM scientist reads `program.md` (research goals) + past results, proposes a modified config
3. **Each experiment:** TOTO fine-tunes on the energy dataset, evaluates, returns metrics
4. **Logging:** Every experiment saved to `experiments/exp_XXX/` with config, metrics, and reasoning
5. **Leaderboard:** `leaderboard.md` regenerated after each run, ranked by MAE

## Project Structure

```
nightshift/
├── run_local.py            # CLI — single machine (recommended)
├── run.py                  # CLI — split mode (local scientist + remote GPU)
├── program.md              # Research goals (read by the SLM scientist)
├── experiment.yaml         # Current config (modified by scientist each iteration)
├── leaderboard.md          # Generated after each run
├── agent/
│   ├── scientist.py        # Ollama SLM proposes experiment configs
│   ├── local_executor.py   # Runs training via subprocess (single machine)
│   ├── executor.py         # Runs training via SSH (split mode)
│   ├── logger.py           # Log experiments, generate leaderboard
│   └── loop.py             # Orchestration: scientist → executor → logger
├── scripts/
│   ├── train_remote.py     # TOTO fine-tune + eval script
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

**proenfo_gfc12** — hourly electricity load from the [FEV Benchmark](https://huggingface.co/datasets/autogluon/fev_datasets). Includes air temperature as an exogenous covariate. Auto-downloaded by TOTO at training time.

## Cost

- **[RunPod](https://runpod.io?ref=de53e8ds) RTX 4000 Ada:** ~$0.26/hr
- **100 experiments overnight (~8 hrs):** ~$2.08
- **Ollama:** Free (runs on same instance)

## Inspired By

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — autonomous AI research agent pattern
- [Datadog TOTO](https://github.com/DataDog/toto) — time series foundation model with exogenous covariates
