#!/usr/bin/env bash
# One-time setup of SymPointV2 on a Linux / WSL2 machine with an NVIDIA GPU (tested target: 16 GB VRAM).
# Usage:  bash setup.sh [install_dir]      (needs conda and the NVIDIA driver; CUDA toolkit 12.x for nvcc)
set -euo pipefail
DIR="${1:-$HOME/sympointv2}"
HERE="$(cd "$(dirname "$0")" && pwd)"
git clone https://github.com/nicehuster/SymPointV2 "$DIR" || true
cd "$DIR"
conda create -y -n sympoint python=3.10
eval "$(conda shell.bash hook)"
conda activate sympoint
# Recent PyTorch: the repository pins torch 1.10 + CUDA 11.1, which does not support RTX 40xx / 50xx GPUs.
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | cut -d. -f1)
if [ "${CC:-8}" -ge 12 ]; then   # RTX 50xx (Blackwell) needs CUDA 12.8 builds
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
else                             # RTX 30xx / 40xx and older
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
fi
pip install munch pyyaml numpy svgpathtools scikit-learn tqdm tensorboardx
# The point ops include <THC/THC.h>, removed from PyTorch >= 1.11 (nothing else from THC is used).
sed -i '/#include <THC\/THC.h>/d' modules/pointops/src/*/*.cpp
(cd modules/pointops && python setup.py install)
cp "$HERE/server.py" "$DIR/sympoint_server.py"
echo
echo "Done. Download the weights (Google Drive link in the SymPointV2 README) to $DIR/weights.pth, then:"
echo "  cd $DIR && conda activate sympoint && python sympoint_server.py --config configs/svg/svg_pointT.yaml --checkpoint weights.pth"
