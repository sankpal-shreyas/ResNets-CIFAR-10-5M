# Efficient ResNets for CIFAR-10 under 5M parameters

NYU Tandon Deep Learning (Fall 2026), Project 1 Kaggle competition
*Efficient ResNet Architecture Challenge* (`cifar-10-res-net-architecture-challenge-dl`).

Goal: the highest accuracy on a hidden CIFAR-10-style test set with a residual network of at most
**5,000,000 trainable parameters**, trained from scratch on the 50,000 CIFAR-10 training images only.

> Results and the final model are filled in as experiments finish (see `scripts/summarize_runs.py`).

## Setup

```bash
uv venv --python 3.12 .venv            # or: python -m venv .venv
uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv -r requirements.txt
```

## Data

```bash
python scripts/download_kaggle_data.py   # kagglehub; token in ~/.kaggle/access_token
python scripts/verify_data.py            # checks the Kaggle training set equals CIFAR-10 train
```

* `kagglehub.competition_download(...)` is tried first. Kaggle has not generated the bundled archive
  for this competition (API answer: 404 "no gcs url"), so the script falls back to downloading the
  files one by one with kagglehub (rate-limited; resumable).
* The competition's `train_images/` + `train_labels.csv` are **identical** to the standard CIFAR-10
  training set (same order, same pixels; checked by `verify_data.py`), so training reads the
  torchvision copy (`data/cifar10`), which loads much faster than 50,000 PNGs.
* Model selection uses a stratified 45,000 / 5,000 split of the training images (`data.val_size`).
  The hidden test images are only ever used to write predictions.

## Training

```bash
python train.py --config configs/wrn28_ta.yaml                       # one run
python train.py --config configs/wrn28_ta.yaml --set train.epochs=50 name=quick   # overrides
bash scripts/run_queue.sh queues/phase1.txt                          # a list of runs
python scripts/summarize_runs.py                                     # results table
```

Each run writes `runs/<name>/{config.yaml, log.csv, model.txt, model_final.pt, summary.json,
val_probs.npy}`. Interrupted runs continue with `--resume`. Every student config asserts the
5M-parameter limit before training starts.

Training features (all configurable, see `resnet5m/config.py` for defaults and descriptions):
random crop + flip, TrivialAugment / AutoAugment / RandAugment, Cutout, MixUp / CutMix, label
smoothing, SGD with Nesterov momentum and cosine schedule, weight EMA, (late-phase) SAM / ASAM,
ShakeDrop / Shake-Shake / stochastic depth, knowledge distillation from teachers trained from
scratch on the same data, and test-time augmentation.

## Predicting the hidden test set

```bash
python predict.py --ckpt runs/<name>/model_final.pt --tta flip_shift --out submission.csv
```

Prints the trainable-parameter count and validates the file against `sample_submission.csv`.

## Large teachers on NYU HPC

See [scripts/hpc/README.md](scripts/hpc/README.md).

## Repository layout

```
train.py, predict.py        entry points
resnet5m/models/            WRN, Shake-Shake, PyramidNet(+ShakeDrop), ResNet; parameter counting
resnet5m/data.py            CIFAR-10 loading, stratified split, augmentation, Kaggle test images
resnet5m/mix.py             Cutout, MixUp, CutMix, soft-target and distillation losses
resnet5m/sam.py, ema.py     SAM/ASAM and weight averaging
resnet5m/inference.py       TTA inference, checkpoint loading
configs/, queues/           experiment configurations
scripts/                    data download/verification, run queue, summaries, HPC jobs
```

## Compliance

* ≤ 5,000,000 trainable parameters (asserted in `train.py` and `predict.py`).
* Trained from scratch on the 50k CIFAR-10 training images only; no pretrained weights or external data.
* Distillation teachers are trained from scratch by this repository on the same training split.
* Hidden test images are never used for training, adaptation, or model selection.

## References

* K. He et al. Deep residual learning (CVPR 2016); Identity mappings in deep residual networks (ECCV 2016).
* S. Zagoruyko, N. Komodakis. Wide Residual Networks (BMVC 2016).
* X. Gastaldi. Shake-Shake regularization (2017); PyTorch port: github.com/hysts/pytorch_shake_shake.
* D. Han et al. Deep Pyramidal Residual Networks (CVPR 2017); github.com/dyhan0920/PyramidNet-PyTorch.
* Y. Yamada et al. ShakeDrop regularization (2018).
* S. Müller, F. Hutter. TrivialAugment (ICCV 2021); E. Cubuk et al. AutoAugment (CVPR 2019), RandAugment (2020).
* T. DeVries, G. Taylor. Cutout (2017); H. Zhang et al. mixup (ICLR 2018); S. Yun et al. CutMix (ICCV 2019).
* P. Foret et al. SAM (ICLR 2021); J. Kwon et al. ASAM (ICML 2021); Z. Zhou et al. SAM late in training (ICLR 2025);
  SAM reference code: github.com/davda54/sam.
* G. Hinton et al. Distilling the knowledge in a neural network (2015); L. Beyer et al. Knowledge distillation:
  a good teacher is patient and consistent (CVPR 2022).
* K. Jordan. 94% on CIFAR-10 in 3.29 seconds on a single GPU (airbench, 2024) — TTA scheme.
* kuangliu/pytorch-cifar — reference CIFAR ResNet training code suggested by the course.
