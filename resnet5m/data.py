"""CIFAR-10 data: loading, train/validation split, CPU-side augmentation, Kaggle test images.

Training data. The competition's train_images/ + train_labels.csv are identical to the standard
CIFAR-10 training set (same order, same pixels; see scripts/verify_data.py), so we read the
torchvision copy, which loads much faster than 50,000 PNG files.

Validation. All model selection uses a stratified hold-out split of the training data
(val_size images, val_size/10 per class); the hidden test images are only ever predicted.
The official CIFAR-10 test batch (10,000 labelled images) is never trained on; it is a second
held-out set, the only one for models trained on all 50k images.
"""
import math
import os

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.v2 as v2
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def load_cifar10_train(root="data"):
    """Return (images uint8 [N,3,32,32] numpy, labels int64 [N] numpy)."""
    ds = torchvision.datasets.CIFAR10(os.path.join(root, "cifar10"), train=True, download=True)
    return np.ascontiguousarray(ds.data.transpose(0, 3, 1, 2)), np.asarray(ds.targets, dtype=np.int64)


def load_cifar10_test(root="data"):
    """The official CIFAR-10 test batch, for evaluation only: (uint8 [10000,3,32,32], int64 [10000])."""
    ds = torchvision.datasets.CIFAR10(os.path.join(root, "cifar10"), train=False, download=True)
    return np.ascontiguousarray(ds.data.transpose(0, 3, 1, 2)), np.asarray(ds.targets, dtype=np.int64)


def stratified_split(labels, val_size, seed=0):
    """Indices (train, val) with val_size // n_classes validation images per class."""
    if val_size <= 0:
        return np.arange(len(labels)), np.array([], dtype=np.int64)
    rng = np.random.RandomState(seed)
    classes = np.unique(labels)
    per_class = val_size // len(classes)
    val = np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:per_class] for c in classes])
    val = np.sort(val)
    train = np.setdiff1d(np.arange(len(labels)), val)
    return train, val


def load_test_images(root="data"):
    """Kaggle hidden-test images in the order of test.csv: (ids int64 [N], uint8 [N,3,32,32])."""
    meta = pd.read_csv(os.path.join(root, "kaggle", "test.csv"))
    imgs = [np.asarray(Image.open(os.path.join(root, "kaggle", "test_images", f)).convert("RGB"))
            for f in meta["filename"]]
    return meta["id"].to_numpy(), np.stack(imgs).transpose(0, 3, 1, 2).copy()


def photometric_ops():
    """Extra low-level variation (colour, contrast, sharpness, compression) on top of the
    auto-augment policy, so the model depends less on CIFAR-10's particular image statistics."""
    sharpness = [v2.RandomAdjustSharpness(f, p=1.0) for f in (0.0, 0.5, 2.0, 3.0)]  # 0 = blur
    return [
        v2.RandomApply([v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)], p=0.5),
        v2.RandomApply([v2.RandomChoice(sharpness)], p=0.3),
        v2.RandomApply([v2.JPEG((50, 95))], p=0.2),
    ]


def build_cpu_transform(policy="ta", pad=4, pad_mode="reflect", photometric=False):
    """Per-sample augmentation on uint8 CHW tensors (runs in DataLoader workers).
    Random crop + horizontal flip, then an automatic policy:
    'ta' TrivialAugmentWide (Mueller & Hutter 2021), 'aa' AutoAugment CIFAR-10 policy
    (Cubuk et al. 2019), 'ra' RandAugment N=2 M=14 (Cubuk et al. 2020), 'none';
    optionally followed by photometric_ops()."""
    ops = []
    if pad > 0:
        ops.append(v2.RandomCrop(32, padding=pad, padding_mode=pad_mode))
    ops.append(v2.RandomHorizontalFlip())
    if policy == "ta":
        ops.append(v2.TrivialAugmentWide())
    elif policy == "aa":
        ops.append(v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10))
    elif policy == "ra":
        ops.append(v2.RandAugment(num_ops=2, magnitude=14))
    elif policy != "none":
        raise ValueError(f"unknown augmentation policy '{policy}'")
    if photometric:
        ops += photometric_ops()
    return v2.Compose(ops)


class AugmentedDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(torch.from_numpy(self.images[i])), self.labels[i]


class RepeatedAugSampler(Sampler):
    """Augmentation multiplicity (Fort et al. 2021): each batch holds `repeats` independently
    augmented copies of batch_size/repeats images; the epoch length stays ~len(dataset)."""

    def __init__(self, n, repeats, seed=0):
        self.n, self.repeats, self.epoch, self.seed = n, repeats, 0, seed

    def __len__(self):
        return math.ceil(self.n / self.repeats) * self.repeats

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        idx = torch.randperm(self.n, generator=g)[: math.ceil(self.n / self.repeats)]
        return iter(idx.repeat_interleave(self.repeats).tolist())


def make_train_loader(images, labels, batch_size, policy="ta", pad=4, num_workers=8,
                      repeats=1, seed=0, photometric=False):
    ds = AugmentedDataset(images, labels, build_cpu_transform(policy, pad, photometric=photometric))
    sampler = RepeatedAugSampler(len(ds), repeats, seed) if repeats > 1 else None
    return DataLoader(ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
                      num_workers=num_workers, persistent_workers=num_workers > 0,
                      pin_memory=True, drop_last=True,
                      prefetch_factor=4 if num_workers > 0 else None)


class GpuNormalize:
    """uint8 NCHW batch on any device -> normalised float (channels_last) on `device`."""

    def __init__(self, device):
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1) * 255
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1) * 255
        self.device = device

    def __call__(self, x):
        x = x.to(self.device, non_blocking=True).float()
        return ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
