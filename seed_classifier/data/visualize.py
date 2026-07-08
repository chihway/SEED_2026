"""Build a grid collage of example cutouts per blend-severity class, for
report figures.

Usage (from the SEED repo root, in the `seed` conda env):
    python -m seed_classifier.data.visualize --data-dir data/sim/run003 \
        --out reports/run003_assets/class_examples.png
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CLASS_NAMES = {1: "Clean / isolated", 2: "Moderate blend", 3: "Severe blend", 4: "Ambiguous overlap"}


def to_rgb(img):
    """(3, H, W) in band order g, r, i -> HxWx3 RGB, i/r/g mapped to R/G/B
    (astronomy convention: redder band to red channel), arcsinh-stretched."""
    g, r, i = img[0], img[1], img[2]
    rgb = np.stack([i, r, g], axis=-1)
    scale = np.percentile(rgb, 99.5)
    scale = scale if scale > 0 else 1.0
    stretched = np.arcsinh(rgb / (scale / 3))
    return np.clip(stretched / stretched.max(), 0, 1) if stretched.max() > 0 else stretched


def make_collage(data_dir, out_path, n_per_class=6, seed=0):
    labels = pd.read_parquet(f"{data_dir}/labels.parquet")
    images = np.load(f"{data_dir}/images.npy", mmap_mode="r")
    rng = np.random.RandomState(seed)

    fig, axes = plt.subplots(4, n_per_class, figsize=(n_per_class * 1.8, 4 * 1.8 + 0.6))
    for row, cls in enumerate([1, 2, 3, 4]):
        idx_pool = labels.index[labels["class"] == cls].to_numpy()
        chosen = rng.choice(idx_pool, size=min(n_per_class, len(idx_pool)), replace=False)
        for col in range(n_per_class):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col < len(chosen):
                img = np.asarray(images[chosen[col]], dtype=np.float32)
                ax.imshow(to_rgb(img))
            else:
                ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"{cls} · {CLASS_NAMES[cls]}", fontsize=10, rotation=90, labelpad=8)

    fig.suptitle("Example cutouts by blend-severity class (g/r/i → RGB)", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-per-class", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    make_collage(args.data_dir, args.out, n_per_class=args.n_per_class, seed=args.seed)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
