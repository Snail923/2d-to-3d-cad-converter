#!/usr/bin/env bash
# One-time setup of the SymPointV2 service on Linux / WSL2 with an NVIDIA GPU.
# Leaves the system CUDA alone: nvcc 12.8 and GCC 13 are installed INSIDE the conda env only.
#
#   bash sympoint_converter/service/setup.sh [install_dir]          (default ~/sympointv2)
#
# Needs: conda (Miniforge recommended), git, an NVIDIA driver that supports CUDA >= 12.8
# (nvidia-smi shows "CUDA Version: 12.8" or higher). Safe to re-run: finished steps are skipped.
set -euo pipefail
DIR="${1:-$HOME/sympointv2}"
ENV=sympoint
HERE="$(cd "$(dirname "$0")" && pwd)"

command -v conda >/dev/null || { echo "conda not found: install Miniforge first (see README.md)"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: install the NVIDIA driver first"; exit 1; }
DRV=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)
if [ -n "$DRV" ] && [ "$DRV" -lt 128 ]; then
  echo "The driver supports CUDA ${DRV:0:2}.${DRV:2}; 12.8 or newer is needed. Update the NVIDIA driver."; exit 1
fi

[ -d "$DIR/.git" ] || git clone https://github.com/nicehuster/SymPointV2 "$DIR"
eval "$(conda shell.bash hook)"
conda env list | grep -qE "^$ENV\s" || conda create -y -n "$ENV" python=3.10
conda activate "$ENV"

echo "== private CUDA 12.8 compiler + GCC 13 (inside the env only)"
[ -x "$CONDA_PREFIX/bin/nvcc" ] || conda install -y -c conda-forge cuda-version=12.8 cuda-nvcc cuda-cudart-dev \
  libcublas-dev libcusparse-dev libcusolver-dev libcurand-dev gcc_linux-64=13 gxx_linux-64=13
conda deactivate && conda activate "$ENV"      # re-run the activation scripts (CC / CXX -> GCC 13)

echo "== PyTorch (CUDA 12.8 build) and SymPointV2 dependencies"
python -c "import torch" 2>/dev/null || pip install --timeout 120 torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install munch pyyaml numpy scipy svgpathtools scikit-learn tqdm tensorboardx

echo "== compile pointops with the private nvcc"
if ! python -c "import torch, pointops_cuda" 2>/dev/null; then
  export CUDA_HOME="$CONDA_PREFIX" PATH="$CONDA_PREFIX/bin:$PATH"
  CC_MAJ=$(python -c "import torch; print(torch.cuda.get_device_capability(0)[0])" 2>/dev/null || echo "")
  CC_MIN=$(python -c "import torch; print(torch.cuda.get_device_capability(0)[1])" 2>/dev/null || echo "")
  [ -n "$CC_MAJ" ] && export TORCH_CUDA_ARCH_LIST="$CC_MAJ.$CC_MIN"
  nvcc --version | tail -1
  # <THC/THC.h> was removed from PyTorch >= 1.11; nothing else from THC is used.
  sed -i '/#include <THC\/THC.h>/d' "$DIR"/modules/pointops/src/*/*.cpp
  (cd "$DIR/modules/pointops" && rm -rf build ./*.egg-info && pip install --no-build-isolation .)
fi
python -c "import torch, pointops_cuda; print('pointops OK on', torch.cuda.get_device_name(0))"

cp "$HERE/server.py" "$DIR/sympoint_server.py"
echo
echo "Done. Put the weights (best.pth) and the .yaml that came with them in $DIR, then:"
echo "  cd $DIR && conda activate $ENV && python sympoint_server.py --config svg_pointT_release.yaml --checkpoint weights.pth"
