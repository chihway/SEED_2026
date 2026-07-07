"""
Render a grid of example cutouts spanning the blendedness range, so class
thresholds in labeling.py can be tuned by eye before generating a full
training set. Saves a PNG grid sorted by blendedness, each panel annotated
with its B value and the class the current default thresholds would assign.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from blend_sim import render_scene, extract_cutouts, load_catalog, BAND_PARAMS  # noqa: E402
from labeling import classify, CLASS_NAMES  # noqa: E402

CATALOG_PATH = "/Users/chihwaychang/Work/descwl_shear_sims/catsim/OneDegSq.fits"
RGB_BANDS = ["i", "r", "g"]  # -> R, G, B channels (astronomy convention)


def asinh_stretch(img, sigma, nonlinearity=3.0):
    scaled = img / (sigma * nonlinearity)
    return np.arcsinh(scaled) / np.arcsinh(1.0 / nonlinearity * 30)


def make_rgb(cutout, bands):
    chans = []
    for b in RGB_BANDS:
        idx = bands.index(b)
        chans.append(asinh_stretch(cutout[idx], BAND_PARAMS[b]["sky_sigma"]))
    rgb = np.stack(chans, axis=-1)
    return np.clip(rgb, 0, 1)


def collect_examples(n_scenes, coadd_dim=250, buff=25, stamp_size=64, seed=0):
    rng = np.random.RandomState(seed)
    catalog = load_catalog(CATALOG_PATH)
    all_results = []
    for s in range(n_scenes):
        scene = render_scene(rng, catalog, coadd_dim=coadd_dim, buff=buff,
                              bands=RGB_BANDS)
        all_results.extend(extract_cutouts(scene, stamp_size=stamp_size))
    return all_results


def pick_examples(results, targets, tol_rank=True):
    """Pick the example closest to each target blendedness value."""
    b_values = np.array([r["blendedness"] for r in results])
    picked = []
    for target in targets:
        idx = int(np.argmin(np.abs(b_values - target)))
        picked.append(results[idx])
    return picked


def render_grid(examples, out_path, ncols=5):
    n = len(examples)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, ex in zip(axes, examples):
        rgb = make_rgb(ex["cutout"], ex["bands"])
        ax.imshow(rgb, origin="lower")
        cls = classify(ex["blendedness"])
        ax.set_title(f"B={ex['blendedness']:.3f}  n_nb={ex['n_sig_neighbors']}\n"
                     f"class {cls}: {CLASS_NAMES[cls]}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(examples):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")


if __name__ == "__main__":
    print("collecting examples across many scenes...")
    results = collect_examples(n_scenes=40)
    print(f"total usable cutouts collected: {len(results)}")
    b_values = np.array([r["blendedness"] for r in results])
    print("blendedness range: min=%.3f max=%.3f" % (b_values.min(), b_values.max()))

    targets = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.2,
               0.3, 0.4, 0.5, 0.6, 0.75, 0.9]
    examples = pick_examples(results, targets)
    examples.sort(key=lambda r: r["blendedness"])

    out_path = "/Users/chihwaychang/Work/SEED_2026/data/sim/calibration_grid.png"
    render_grid(examples, out_path)
