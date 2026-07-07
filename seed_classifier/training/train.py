"""Train the ordinal CNN blend-severity classifier on a dataset run directory
produced by simulation/generate_dataset.py.

Usage (from the SEED repo root, in the `seed` conda env):
    python -m seed_classifier.training.train --data-dir data/sim/test_run --epochs 30
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from seed_classifier.data.dataset import (
    BlendCutoutDataset, class_sample_weights, compute_norm_stats, scene_split,
)
from seed_classifier.models.cnn import BlendCNN
from seed_classifier.models.ordinal import corn_loss, corn_predict

NUM_CLASSES = 4


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = corn_loss(logits, labels, NUM_CLASSES)
            total_loss += loss.item() * images.size(0)
            n += images.size(0)
            all_preds.append(corn_predict(logits).cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    qwk = cohen_kappa_score(targets, preds, weights="quadratic")
    bal_acc = balanced_accuracy_score(targets, preds)
    return total_loss / n, qwk, bal_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", default=None, help="defaults to <repo-root>/models/<basename(data-dir)>")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"using device: {device}")

    labels = pd.read_parquet(os.path.join(args.data_dir, "labels.parquet"))
    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    n_bands = len(meta["bands"])

    train_idx, val_idx = scene_split(labels, val_frac=args.val_frac, seed=args.seed)
    stats = compute_norm_stats(args.data_dir, train_idx, seed=args.seed)

    train_ds = BlendCutoutDataset(args.data_dir, indices=train_idx, stats=stats, augment=True)
    val_ds = BlendCutoutDataset(args.data_dir, indices=val_idx, stats=stats)

    sample_weights = class_sample_weights(labels, train_idx)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = BlendCNN(n_bands=n_bands, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "models", os.path.basename(os.path.normpath(args.data_dir)),
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "norm_stats.json"), "w") as f:
        json.dump({k: v.tolist() for k, v in stats.items()}, f, indent=2)

    best_qwk = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, n = 0.0, 0
        for images, labels_batch in train_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = corn_loss(logits, labels_batch, NUM_CLASSES)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            n += images.size(0)
        scheduler.step()

        val_loss, val_qwk, val_bal_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch:3d}  lr {scheduler.get_last_lr()[0]:.5f}  train_loss {running_loss / n:.4f}  "
              f"val_loss {val_loss:.4f}  val_qwk {val_qwk:.4f}  val_bal_acc {val_bal_acc:.4f}")

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))

    print(f"\nbest val QWK: {best_qwk:.4f}")


if __name__ == "__main__":
    main()
