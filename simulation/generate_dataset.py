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


def generate(n_scenes, out_dir, coadd_dim=250, buff=25, stamp_size=64, seed=0,
             cutouts_per_scene_estimate=18):
    """
    Streams cutouts straight to a preallocated memmap'd images.npy instead of
    collecting them in a Python list and np.stack-ing at the end. At full
    scale (10k+ scenes, 100k+ cutouts) the old list-then-stack approach needs
    ~2x the final dataset size in RAM simultaneously (the list of arrays plus
    the freshly stacked copy) -- on this machine that reliably got the process
    OOM-killed right at the finish line, after 45+ minutes of simulation work.
    Streaming keeps peak memory at O(1 scene) regardless of dataset size.

    `cutouts_per_scene_estimate` sizes the preallocated array (with headroom)
    since the exact count isn't known until all scenes are processed; the
    file is trimmed down to the true count at the end.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    catalog = load_catalog(CATALOG_PATH)
    n_bands = len(BANDS)

    images_path = os.path.join(out_dir, "images.npy")
    capacity = n_scenes * cutouts_per_scene_estimate
    images = np.lib.format.open_memmap(
        images_path, mode="w+", dtype=np.float32,
        shape=(capacity, n_bands, stamp_size, stamp_size),
    )

    rows = []
    n_written = 0
    for scene_id in range(n_scenes):
        scene = render_scene(rng, catalog, coadd_dim=coadd_dim, buff=buff, bands=BANDS)
        for r in extract_cutouts(scene, stamp_size=stamp_size):
            if n_written >= capacity:
                raise RuntimeError(
                    f"cutouts_per_scene_estimate={cutouts_per_scene_estimate} was too "
                    f"low for this run -- exceeded preallocated capacity {capacity}"
                )
            images[n_written] = r["cutout"].astype(np.float32)
            rows.append({
                "scene_id": scene_id,
                "obj_index": r["obj_index"],
                "blendedness": r["blendedness"],
                "n_sig_neighbors": r["n_sig_neighbors"],
                "sep_nearest_arcsec": r["sep_nearest_arcsec"],
                "flux_ratio_nearest": r["flux_ratio_nearest"],
                "redshift": r["redshift"],
                "r_ab": r["r_ab"],
                "local_dominance": r["local_dominance"],
                "class": classify(r["blendedness"]),
            })
            n_written += 1
        if (scene_id + 1) % 50 == 0:
            print(f"  {scene_id + 1}/{n_scenes} scenes done, {n_written} cutouts so far")

    images.flush()
    del images  # close the oversized memmap before trimming

    _trim_memmap(images_path, n_written)

    labels = pd.DataFrame(rows)
    labels["cutout_index"] = np.arange(len(labels))

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

    return labels


def _trim_memmap(path, n_actual, chunk=2000):
    """Shrink a preallocated images.npy down to its true row count, copying in
    chunks so peak memory stays at O(chunk) rather than O(full dataset)."""
    oversized = np.lib.format.open_memmap(path, mode="r")
    tmp_path = path + ".tmp"
    trimmed = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=oversized.dtype, shape=(n_actual,) + oversized.shape[1:]
    )
    for start in range(0, n_actual, chunk):
        end = min(start + chunk, n_actual)
        trimmed[start:end] = oversized[start:end]
    trimmed.flush()
    del trimmed
    del oversized
    os.replace(tmp_path, path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-scenes", type=int, default=500)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    generate(args.n_scenes, args.out_dir, seed=args.seed)
