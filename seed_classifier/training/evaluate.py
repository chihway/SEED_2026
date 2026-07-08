"""Evaluate a trained checkpoint on its held-out validation split and write
report-ready assets (confusion matrix, classification report, class
distribution, eval_summary.json) to an output directory.

Usage (from the SEED repo root, in the `seed` conda env):
    python -m seed_classifier.training.evaluate --data-dir data/sim/run003 \
        --model-dir models/run003 --out-dir reports/run003_assets
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score, classification_report, cohen_kappa_score, confusion_matrix,
)
from torch.utils.data import DataLoader

from seed_classifier.data.dataset import BlendCutoutDataset, scene_split
from seed_classifier.models.cnn import BlendCNN
from seed_classifier.models.ordinal import corn_predict

NUM_CLASSES = 4
CLASS_NAMES = ["clean", "moderate", "severe", "ambiguous"]


def run_inference(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            all_preds.append(corn_predict(logits).cpu().numpy())
            all_targets.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def plot_confusion_matrix(cm, out_path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    thresh = cm.max() / 2
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_class_distribution(labels_df, out_path):
    counts = labels_df["class"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(CLASS_NAMES, [counts.get(i, 0) for i in range(1, NUM_CLASSES + 1)], color="#4C72B0")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    for i, name in enumerate(CLASS_NAMES):
        ax.text(i, counts.get(i + 1, 0), f"{counts.get(i + 1, 0):,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    labels = pd.read_parquet(os.path.join(args.data_dir, "labels.parquet"))
    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    n_bands = len(meta["bands"])

    _, val_idx = scene_split(labels, val_frac=args.val_frac, seed=args.seed)

    with open(os.path.join(args.model_dir, "norm_stats.json")) as f:
        raw_stats = json.load(f)
    stats = {k: np.array(v) for k, v in raw_stats.items()}

    val_ds = BlendCutoutDataset(args.data_dir, indices=val_idx, stats=stats)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = BlendCNN(n_bands=n_bands, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "best_model.pt"), map_location=device))

    preds, targets = run_inference(model, val_loader, device)

    qwk = cohen_kappa_score(targets, preds, weights="quadratic")
    bal_acc = balanced_accuracy_score(targets, preds)
    cm = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))
    report_text = classification_report(targets, preds, target_names=CLASS_NAMES, digits=2)
    report_dict = classification_report(targets, preds, target_names=CLASS_NAMES, digits=2, output_dict=True)

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "eval_summary.json"), "w") as f:
        json.dump({
            "qwk": qwk,
            "bal_acc": bal_acc,
            "cm": cm.tolist(),
            "n_val": int(len(val_idx)),
            "per_class_f1": {name: report_dict[name]["f1-score"] for name in CLASS_NAMES},
        }, f, indent=2)

    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write(report_text)

    plot_confusion_matrix(cm, os.path.join(args.out_dir, "confusion_matrix.png"))
    plot_class_distribution(labels, os.path.join(args.out_dir, "class_distribution.png"))

    print(f"val QWK: {qwk:.4f}  balanced accuracy: {bal_acc:.4f}  n_val: {len(val_idx)}")
    print(report_text)
    print(f"assets written to {args.out_dir}")


if __name__ == "__main__":
    main()
