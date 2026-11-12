#!/bin/bash
# One-time environment setup on the NYU burst cluster. Run it in a terminal of a CPU-only
# Jupyter session (partition 'interactive'): GPU instances that sit idle for 20 minutes are
# shut down, and this needs no GPU. Everything goes under WORK (default /scratch/$USER/dl-p1):
#   venv/   Python 3.12 + PyTorch 2.14.1 (CUDA 12.6 wheels: V100, A100, L4, H100)
#   data/   kaggle/ from data.zip (train + test images), or torchvision's CIFAR-10 training set
# Usage:  bash scripts/hpc/setup_env.sh        (from the unpacked code directory)
set -euo pipefail
cd "$(dirname "$0")/../.."   # code root, so 'import resnet5m' works below
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

# Training data. Preferred: the Kaggle competition zip (data.zip, uploaded to $HOME or $WORK),
# unpacked to $WORK/data/kaggle (also holds the test images predict.py needs). The 50k training
# PNGs are then read once here and cached as one .npz, so GPU jobs start training immediately
# and parallel jobs never race to build the cache. Without data.zip: torchvision's CIFAR-10
# download (slow from cs.toronto.edu; the training images are identical).
KAGGLE="$WORK/data/kaggle"
if [ ! -f "$KAGGLE/train_labels.csv" ]; then
  for z in "$WORK/data.zip" "$HOME/data.zip"; do
    if [ -f "$z" ]; then
      echo "unpacking $z -> $KAGGLE (52k small files, a few minutes)"
      mkdir -p "$KAGGLE"
      unzip -q -o "$z" -d "$KAGGLE" 2>/dev/null || "$WORK/venv/bin/python" -m zipfile -e "$z" "$KAGGLE"
      break
    fi
  done
fi
if [ -f "$KAGGLE/train_labels.csv" ]; then
  echo "train images: $(ls "$KAGGLE/train_images" | wc -l)  test images: $(ls "$KAGGLE/test_images" | wc -l)"
  "$WORK/venv/bin/python" -c "from resnet5m.data import load_cifar10_train as f; x, y = f('$WORK/data'); print('training cache ok:', x.shape, y.shape)"
else
  echo "no data.zip found in \$HOME or $WORK: downloading CIFAR-10 with torchvision instead"
  "$WORK/venv/bin/python" -c "import torchvision; torchvision.datasets.CIFAR10('$WORK/data/cifar10', train=True, download=True)"
fi

"$WORK/venv/bin/python" -c "import torch; print('torch', torch.__version__, '| GPU visible here:', torch.cuda.is_available(), '(False is expected on a CPU Jupyter node)')"
uv cache clean >/dev/null 2>&1 || true   # free ~3 GB and many inodes on /scratch
echo "setup done: venv=$WORK/venv data=$WORK/data"
