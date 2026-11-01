#!/bin/bash
# One-time environment setup on the NYU burst cluster. Run it in a terminal of a CPU-only
# Jupyter session (partition 'interactive'): GPU instances that sit idle for 20 minutes are
# shut down, and this needs no GPU. Everything goes under WORK (default /scratch/$USER/dl-p1):
#   venv/   Python 3.12 + PyTorch 2.14.1 (CUDA 12.6 wheels: V100, A100, L4, H100)
#   data/   CIFAR-10 training set (identical to the competition's train_images + train_labels.csv)
# Usage:  bash scripts/hpc/setup_env.sh        (from the unpacked code directory)
set -euo pipefail
WORK=${WORK:-/scratch/$USER/dl-p1}
mkdir -p "$WORK"
# keep uv's download cache (~3 GB) and its Python install off the small home quota
export UV_CACHE_DIR="$WORK/.uv-cache" UV_PYTHON_INSTALL_DIR="$WORK/.uv-python"

if "$WORK/venv/bin/python" -c "import torch, torchvision, numpy, pandas, PIL, yaml" 2>/dev/null; then
  echo "existing environment in $WORK/venv works: skipping the install"
else
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  rm -rf "$WORK/venv"   # start clean if an earlier attempt left a partial environment
  uv venv --python 3.12 "$WORK/venv"
  uv pip install --python "$WORK/venv/bin/python" "torch==2.14.1" torchvision \
    --index-url https://download.pytorch.org/whl/cu126
  uv pip install --python "$WORK/venv/bin/python" numpy pandas pillow pyyaml
fi

# If this download is slow, upload data/cifar10/cifar-10-python.tar.gz from the laptop into
# $WORK/data/cifar10/ and rerun: torchvision then only verifies and extracts it.
"$WORK/venv/bin/python" -c "import torchvision; torchvision.datasets.CIFAR10('$WORK/data/cifar10', train=True, download=True)"

"$WORK/venv/bin/python" -c "import torch; print('torch', torch.__version__, '| GPU visible here:', torch.cuda.is_available(), '(False is expected on a CPU Jupyter node)')"
uv cache clean >/dev/null 2>&1 || true   # free ~3 GB and many inodes on /scratch
echo "setup done: venv=$WORK/venv data=$WORK/data"
