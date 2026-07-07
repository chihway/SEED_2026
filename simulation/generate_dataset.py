"""
Generate a full labeled training set: run many simulated scenes, extract
per-galaxy cutouts + truth-based blend metrics, assign the 4-class label,
and save everything to disk for the (separate) PyTorch training pipeline.

Usage:
    python generate_dataset.py --n-scenes 500 --out-dir ../data/sim/run001
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from blend_sim import render_scene, extract_cutouts, load_catalog
from labeling import classify, DEFAULT_THRESHOLDS

CATALOG_PATH = "/Users/chihwaychang/Work/descwl_shear_sims/catsim/OneDegSq.fits"
BANDS = ["g", "r", "i"]


def generate(n_scenes, out_dir, coadd_dim=250, buff=25, stamp_size=64, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    catalog = load_catalog(CATALOG_PATH)

    cutouts = []
    rows = []
    for scene_id in range(n_scenes):
        scene = render_scene(rng, catalog, coadd_dim=coadd_dim, buff=buff, bands=BANDS)
        for r in extract_cutouts(scene, stamp_size=stamp_size):
            cutouts.append(r["cutout"].astype(np.float32))
            rows.append({
                "scene_id": scene_id,
                "obj_index": r["obj_index"],
                "blendedness": r["blendedness"],
                "n_sig_neighbors": r["n_sig_neighbors"],
                "sep_nearest_arcsec": r["sep_nearest_arcsec"],
                "flux_ratio_nearest": r["flux_ratio_nearest"],
                "redshift": r["redshift"],
                "r_ab": r["r_ab"],
                "class": classify(r["blendedness"]),
            })
        if (scene_id + 1) % 50 == 0:
            print(f"  {scene_id + 1}/{n_scenes} scenes done, {len(rows)} cutouts so far")

    images = np.stack(cutouts, axis=0)  # (N, n_bands, stamp, stamp)
    labels = pd.DataFrame(rows)
    labels["cutout_index"] = np.arange(len(labels))

    np.save(os.path.join(out_dir, "images.npy"), images)
    labels.to_parquet(os.path.join(out_dir, "labels.parquet"), index=False)

    meta = {
        "bands": BANDS,
        "coadd_dim": coadd_dim,
        "buff": buff,
        "stamp_size": stamp_size,
        "n_scenes": n_scenes,
        "seed": seed,
        "thresholds": DEFAULT_THRESHOLDS,
        "n_cutouts": len(labels),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nsaved {len(labels)} cutouts to {out_dir}")
    print("class distribution:")
    print(labels["class"].value_counts().sort_index())

    return images, labels


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-scenes", type=int, default=500)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    generate(args.n_scenes, args.out_dir, seed=args.seed)
