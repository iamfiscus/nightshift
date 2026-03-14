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

echo "=== Setup complete ==="
