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
notebooks/             notebooks/01_intro_classifier.ipynb — start here if you're new
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
pip install jupyterlab ipykernel  # only needed for notebooks/
python -m ipykernel install --user --name seed --display-name "Python (seed)"
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

## New to the project? Start with the notebook

[`notebooks/01_intro_classifier.ipynb`](notebooks/01_intro_classifier.ipynb) builds a
classifier from scratch, step by step, on the small `run001` dataset — data loading,
a configurable CNN, a plain training loop, and evaluation, all in one place with a
single editable settings cell. It's a deliberately simplified version of the pipeline
below (plain cross-entropy instead of the CORN ordinal loss) so the fundamentals of
"build a network, train it, read the results" aren't buried under project-specific
complexity. Once it makes sense, `seed_classifier/training/train.py` is the same idea
with the ordinal loss and the full pipeline.

```bash
conda activate seed
cd notebooks
jupyter lab   # open 01_intro_classifier.ipynb, select the "Python (seed)" kernel
```

## Current status

Three runs so far, scaling scene count each time:

| run | scenes | cutouts | QWK | balanced acc |
|---|---|---|---|---|
| run001 | 500 | 6,013 | 0.40 | 0.373 |
| run002 | 3,000 | 36,106 | 0.64 | 0.509 |
| run003 | 10,000 | 114,418 | 0.57 | 0.443 |

run001 -> run002 confirmed that the moderate/ambiguous classes were data-limited: 6x
more scenes roughly doubled their F1. run003 added 3.3x more scenes on top of a fix to
`extract_cutouts`'s detectability check (a target fully swamped by a bright neighbor no
longer counts as a genuine blend). That fix turned out to remove far more of the
**ambiguous** class than expected — its per-scene rate dropped ~15x — leaving it
critically data-starved (153 examples total) and dragging QWK down despite clean and
moderate both improving. **severe** stayed essentially flat across both scale-ups
(F1 0.23 -> 0.31 -> 0.30), suggesting it's intrinsically hard to separate from its
neighbors rather than simply short on data.

Full details: [`reports/session_report.pdf`](reports/session_report.pdf) (run001),
[`reports/run002_report.pdf`](reports/run002_report.pdf),
[`reports/run003_report.pdf`](reports/run003_report.pdf).

**Not yet done:**
- Investigate the ambiguous-class collapse in run003: confirm by eye that the
  now-filtered objects are genuinely swamped-target artifacts, not over-filtering
- Recover ambiguous-class volume (at run003's rate, ~50,000 scenes needed to match
  run002's absolute count) or revisit the detectability threshold
- Revisit the moderate/severe blendedness thresholds in `labeling.py` — severe's
  plateau across two scale-ups makes this more urgent
- Validate against the real LSST+Euclid cutouts (43 usable examples, not yet touched)
