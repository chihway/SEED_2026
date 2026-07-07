"""Quick smoke test: generate one scene, extract cutouts, sanity-check output."""
import numpy as np

from blend_sim import render_scene, extract_cutouts, load_catalog

CATALOG_PATH = "/Users/chihwaychang/Work/descwl_shear_sims/catsim/OneDegSq.fits"

rng = np.random.RandomState(42)
catalog = load_catalog(CATALOG_PATH)
print("catalog size:", len(catalog))

scene = render_scene(rng, catalog, coadd_dim=250, buff=25, bands=["g", "r", "i"])
print("n objects in scene:", len(scene["per_object_truth_images"]))

results = extract_cutouts(scene, stamp_size=64)
print("n usable cutouts:", len(results))

b_values = np.array([r["blendedness"] for r in results])
print("blendedness stats: min=%.3f max=%.3f mean=%.3f median=%.3f" % (
    b_values.min(), b_values.max(), b_values.mean(), np.median(b_values)
))
print("n_sig_neighbors distribution:", np.bincount([r["n_sig_neighbors"] for r in results]))
print("example cutout shape:", results[0]["cutout"].shape)
