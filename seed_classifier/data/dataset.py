"""PyTorch Dataset for SEED blend-severity cutouts produced by
simulation/generate_dataset.py (images.npy + labels.parquet + meta.json).
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BlendCutoutDataset(Dataset):
    """`indices` select rows shared by images.npy (position) and labels.parquet
    (row order matches, since generate_dataset.py appends both in lockstep).
    `stats` must come from `compute_norm_stats` on the *training* split, so
    train/val/test all normalize against the same reference (no leakage).
    """

    def __init__(self, run_dir, indices=None, stats=None, augment=False):
        self.images = np.load(os.path.join(run_dir, "images.npy"), mmap_mode="r")
        self.labels = pd.read_parquet(os.path.join(run_dir, "labels.parquet"))
        with open(os.path.join(run_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.indices = np.arange(len(self.labels)) if indices is None else np.asarray(indices)
        self.stats = stats
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        image = np.asarray(self.images[idx], dtype=np.float32)
        if self.stats is not None:
            image = normalize(image, self.stats)
        if self.augment:
            image = random_dihedral(image)
        label = int(self.labels.iloc[idx]["class"]) - 1  # 1..4 -> 0..3 for CORN
        return torch.from_numpy(image), label


def random_dihedral(image):
    """Random flip + 90-degree rotation. Galaxy orientation on sky is arbitrary,
    so this is a lossless invariance, not a lossy augmentation -- and it matters
    here because the weighted sampler repeats the same handful of rare-class
    (severe/ambiguous) images many times per epoch; without this they were
    getting memorized pixel-for-pixel within a few epochs."""
    if np.random.rand() < 0.5:
        image = image[:, :, ::-1]
    if np.random.rand() < 0.5:
        image = image[:, ::-1, :]
    k = np.random.randint(4)
    if k:
        image = np.rot90(image, k, axes=(1, 2))
    return np.ascontiguousarray(image)


def normalize(image, stats):
    """Per-band arcsinh stretch (robust to the noise-level negative pixels
    every cutout has, unlike log) then standardize with train-split stats."""
    scale = stats["scale"]
    mean = stats["mean"]
    std = stats["std"]
    out = np.arcsinh(image / scale[:, None, None])
    out = (out - mean[:, None, None]) / std[:, None, None]
    return out.astype(np.float32)


def compute_norm_stats(run_dir, indices, sample_size=2000, seed=0):
    """Per-band asinh scale (robust MAD-based noise sigma estimate) plus the
    post-stretch mean/std, from a random sample of the given (training) indices."""
    images = np.load(os.path.join(run_dir, "images.npy"), mmap_mode="r")
    rng = np.random.RandomState(seed)
    sample_idx = np.sort(rng.choice(indices, size=min(sample_size, len(indices)), replace=False))
    sample = np.asarray(images[sample_idx], dtype=np.float32)  # (n, bands, H, W)
    n_bands = sample.shape[1]

    scale = np.array([
        1.4826 * np.median(np.abs(sample[:, b] - np.median(sample[:, b])))
        for b in range(n_bands)
    ])
    scale = np.where(scale > 0, scale, 1.0)

    stretched = np.arcsinh(sample / scale[None, :, None, None])
    mean = stretched.mean(axis=(0, 2, 3))
    std = stretched.std(axis=(0, 2, 3))
    std = np.where(std > 0, std, 1.0)
    return {"scale": scale, "mean": mean, "std": std}


def scene_split(labels_df, val_frac=0.2, seed=0):
    """Split by scene_id, not by row: cutouts from the same simulated scene
    share a sky-noise realization, so splitting rows independently would leak."""
    scene_ids = labels_df["scene_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(scene_ids)
    n_val = max(1, int(len(scene_ids) * val_frac))
    val_scenes = set(scene_ids[:n_val])
    val_mask = labels_df["scene_id"].isin(val_scenes)
    train_idx = labels_df.index[~val_mask].to_numpy()
    val_idx = labels_df.index[val_mask].to_numpy()
    return train_idx, val_idx


def class_sample_weights(labels_df, indices):
    """Inverse-frequency per-sample weights for WeightedRandomSampler, since
    classes are heavily imbalanced (clean >> moderate > severe > ambiguous)."""
    classes = labels_df.iloc[indices]["class"].to_numpy()
    counts = np.bincount(classes, minlength=5)[1:5]  # class labels are 1..4
    class_weight = 1.0 / np.maximum(counts, 1)
    return class_weight[classes - 1]
