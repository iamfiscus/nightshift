#!/bin/bash
# setup_runpod.sh — Run this ON the RunPod instance to install TOTO.
# Usage: bash setup_runpod.sh

set -e

echo "=== Nightshift RunPod Setup ==="

# Install TOTO from source
cd /workspace
if [ ! -d "toto" ]; then
    echo "Cloning TOTO..."
    git clone https://github.com/DataDog/toto.git
fi

cd toto
echo "Installing TOTO dependencies..."
pip install -r requirements.txt
pip install -e .

# Create nightshift working directory
mkdir -p /workspace/nightshift

# Verify TOTO is importable
python -c "from toto.scripts import finetune_toto; print('TOTO installed successfully')"

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Install Ollama
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Clone nightshift repo
cd /workspace
if [ ! -d "nightshift" ]; then
    echo "Cloning nightshift..."
    git clone https://github.com/iamfiscus/nightshift.git
fi

cd nightshift
pip install pyyaml rich paramiko ollama tabulate

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Start Ollama:     ollama serve &"
echo "  2. Pull model:       ollama pull granite4:small-h"
echo "  3. Test (3 runs):    python run_local.py --model granite4:small-h --num 3"
echo "  4. Run overnight:    nohup python run_local.py --model granite4:small-h --num 100 > nightshift.log 2>&1 &"
