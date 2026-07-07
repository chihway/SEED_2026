# SEED_2026

An ordinal CNN that classifies LSST galaxy blend severity from image cutouts, trained
on GalSim simulations and (eventually) validated against real LSST+Euclid data.

Each detected galaxy is classified into one of 4 ordinal severity classes:

1. **Clean / isolated** — no detectable neighbor flux in the footprint
2. **Moderate blend** — a neighbor overlaps but is separable
3. **Severe blend** — clear overlap, likely unusable for shape measurement
4. **Ambiguous overlap** — unclear whether this is 1 or 2 objects

Classes are treated as ordinal (not plain multiclass), since a clean/ambiguous
confusion is a much worse mistake than a clean/moderate one. See
[`reports/session_report.pdf`](reports/session_report.pdf) for the fullest current
writeup: pipeline, architecture, training results, and a confusion matrix.

## Pipeline

| stage | code | what it does |
|---|---|---|
| 1. Simulate scene | `simulation/blend_sim.py` | GalSim + the CatSim catalog render a full random field at real sky density (no `descwl_shear_sims`/`lsst.afw` dependency — see note below) |
| 2. Auto-label | `simulation/labeling.py` | A truth-based blendedness metric *B* (Bosch et al. 2018 style, from noiseless per-object renders) maps each detected object to one of the 4 classes |
| 3. Dataset / DataLoader | `seed_classifier/data/dataset.py` | Loads cutouts, per-band arcsinh-stretch normalization, splits by `scene_id` (not by row, to avoid noise-realization leakage), inverse-frequency sampling for the rare classes, dihedral augmentation |
| 4. Train | `seed_classifier/models/`, `seed_classifier/training/train.py` | Small CNN backbone + [CORN](https://arxiv.org/abs/2111.08851) ordinal head, AdamW + gradient clipping + cosine LR decay, tracked with quadratic weighted kappa (QWK) |

`simulation/generate_dataset.py` runs stages 1–2 end to end and writes a
`images.npy` / `labels.parquet` / `meta.json` triple to a run directory (e.g.
`data/sim/run001`) that `seed_classifier` reads.

> **Why not `descwl_shear_sims`?** It pulls in a 2020, Intel-only LSST DM stack build
> that segfaults under Rosetta on Apple Silicon. `simulation/blend_sim.py` is a
> from-scratch GalSim renderer that sources galaxy morphology/flux directly from the
> CatSim catalog instead, with no `lsst.afw` dependency.

## Repo layout

```
simulation/           GalSim scene renderer, auto-labeling, dataset generation
seed_classifier/
  data/dataset.py      BlendCutoutDataset, scene_split, normalization, class weights
  models/cnn.py         CNN backbone
  models/ordinal.py     CORN ordinal loss + prediction
  training/train.py     training loop (QWK + balanced accuracy, checkpointing)
reports/               session writeups (see session_report.pdf)
data/, models/         generated outputs — gitignored, not checked in
```

## Setup

Two conda environments, kept separate so the simulation stack (GalSim/astropy) never
has to coexist with PyTorch:

- **Simulation** — the `base` conda env, with `galsim` (2.8.4+) and `astropy`.
- **Training** — a dedicated `seed` env (Python 3.11): `torch` 2.2.2, `numpy<2` (pinned
  for ABI compatibility with that torch build), `pandas`, `pyarrow`, `scikit-learn`,
  `matplotlib`, `astropy`.

```bash
conda create -n seed python=3.11
conda activate seed
pip install torch==2.2.2 "numpy<2" pandas pyarrow scikit-learn matplotlib astropy
```

`simulation/blend_sim.py` reads a CatSim catalog FITS file — update `CATALOG_PATH` in
`simulation/generate_dataset.py` to point at your copy of `OneDegSq.fits` (or
equivalent) if it's not at the path currently hardcoded there.

## Usage

Generate a labeled dataset (run in the simulation env):

```bash
conda activate base
cd simulation
python generate_dataset.py --n-scenes 500 --out-dir ../data/sim/run001
```

Train the classifier (run in the `seed` env, from the repo root):

```bash
conda activate seed
python -m seed_classifier.training.train --data-dir data/sim/run001 --epochs 40
```

Checkpoints and normalization stats are written to `models/<run-name>/`.

## Current status

Trained on 500 scenes / 6,013 cutouts. Best validation QWK: **0.40** (moderate
agreement), balanced accuracy 0.373. The model reliably separates clean from
ambiguous (near-zero confusion between the two extremes) but still confuses the
moderate/severe middle classes with each other — plausibly a data-volume issue as much
as a modeling one, since those classes have only ~300–500 examples each in this run.
Full details, plots, and the training-stability debugging story are in
[`reports/session_report.pdf`](reports/session_report.pdf).

**Not yet done:**
- Generate a larger dataset (more scenes) to address the moderate/severe data-volume gap
- Validate against the real LSST+Euclid cutouts (43 usable examples, not yet touched)
- Revisit the moderate/severe blendedness threshold specifically
