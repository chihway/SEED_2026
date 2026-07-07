"""
Generate realistic-ish LSST blended-scene simulations directly with GalSim,
sourcing galaxy morphology/flux from the real CatSim catalog (OneDegSq.fits),
and extract per-galaxy cutouts with a truth-based blendedness metric computed
from noiseless per-object renders.

Deliberately avoids descwl_shear_sims/descwl/lsst.afw: on this machine those
pull in a 2020, Intel-only ("stackvana-afw") build of the LSST DM stack that
segfaults under Rosetta on modern macOS/arm64 (crash isolated to
lsst::geom::operator<< inside descwl.survey.Survey()). This module only needs
galsim + numpy + astropy, all working natively.

Runs in any modern Python env with galsim + astropy installed.
"""
import numpy as np
import galsim
from astropy.io import fits

SCALE = 0.2       # arcsec/pixel, LSST
PSF_FWHM = 0.8    # arcsec, typical seeing (Gaussian approximation)
RANDOM_DENSITY = 60  # galaxies per sq arcmin (tunable; ~LSST detected density)

TRUTH_BAND = "r"

ZEROPOINT = 30.0  # AB mag giving 1 count, same convention for every band

# Approximate LSST 10-year coadd 5-sigma point-source depths (AB mag),
# Ivezic et al. 2019 Table 2. sky_sigma below is *derived* from these via
# the matched-filter relation for a Gaussian PSF:
#   SNR = flux / (sky_sigma * sqrt(4*pi*sigma_psf_pix^2))
# so that a point source at exactly the quoted depth has SNR=5 in that band.
# This keeps the noise/zeropoint pairing self-consistent (unlike guessing
# both independently, which earlier gave implausible peak-pixel S/N for
# every galaxy in the catalog -- see extract_cutouts' detection cut).
_M5_DEPTH = {"u": 25.6, "g": 26.9, "r": 26.9, "i": 26.4, "z": 25.6, "y": 24.7}
_SIGMA_PSF_PIX = (PSF_FWHM / 2.3548) / SCALE
_PSF_AP_FACTOR = np.sqrt(4 * np.pi) * _SIGMA_PSF_PIX


def _sky_sigma_for_depth(m5_mag, snr=5.0):
    flux_at_depth = 10.0 ** (-0.4 * (m5_mag - ZEROPOINT))
    return flux_at_depth / (snr * _PSF_AP_FACTOR)


BAND_PARAMS = {
    b: dict(zeropoint=ZEROPOINT, sky_sigma=_sky_sigma_for_depth(m5))
    for b, m5 in _M5_DEPTH.items()
}

MAG_COLS = {b: f"{b}_ab" for b in BAND_PARAMS}

_catalog_cache = {}


def load_catalog(path):
    if path not in _catalog_cache:
        with fits.open(path) as hdul:
            _catalog_cache[path] = np.array(hdul[1].data)
    return _catalog_cache[path]


def mag_to_flux(mag, zeropoint):
    return 10.0 ** (-0.4 * (mag - zeropoint))


def make_galaxy_profile(row, band):
    """Build a (pre-shift, pre-PSF) GalSim profile for one catalog row/band."""
    zp = BAND_PARAMS[band]["zeropoint"]
    total_flux = mag_to_flux(row[MAG_COLS[band]], zp)

    # fluxnorm_bulge/disk/agn are in some CatSim-internal absolute flux unit
    # that has nothing to do with our chosen zeropoint (they don't sum to 1,
    # e.g. can be ~1e-18) -- only their *relative* split is meaningful here,
    # so normalize them and scale by the AB-magnitude-derived total_flux
    # instead of using their absolute values directly.
    norm_sum = row["fluxnorm_bulge"] + row["fluxnorm_disk"] + row["fluxnorm_agn"]
    bulge_frac = row["fluxnorm_bulge"] / norm_sum if norm_sum > 0 else 0.0
    disk_frac = row["fluxnorm_disk"] / norm_sum if norm_sum > 0 else 0.0
    agn_frac = row["fluxnorm_agn"] / norm_sum if norm_sum > 0 else 0.0

    components = []

    if bulge_frac > 0:
        hlr_b = np.sqrt(row["a_b"] * row["b_b"])
        q_b = row["b_b"] / row["a_b"]
        bulge = galsim.Sersic(n=4, half_light_radius=hlr_b,
                               flux=total_flux * bulge_frac)
        bulge = bulge.shear(galsim.Shear(q=q_b, beta=row["pa_bulge"] * galsim.degrees))
        components.append(bulge)

    if disk_frac > 0:
        hlr_d = np.sqrt(row["a_d"] * row["b_d"])
        q_d = row["b_d"] / row["a_d"]
        disk = galsim.Sersic(n=1, half_light_radius=hlr_d,
                              flux=total_flux * disk_frac)
        disk = disk.shear(galsim.Shear(q=q_d, beta=row["pa_disk"] * galsim.degrees))
        components.append(disk)

    if agn_frac > 0:
        agn = galsim.Gaussian(sigma=1.0e-4, flux=total_flux * agn_frac)
        components.append(agn)

    if not components:
        # fall back: pure exponential disk, shouldn't normally happen
        components.append(galsim.Exponential(half_light_radius=0.3, flux=total_flux))

    return galsim.Add(components)


def sample_scene_catalog(rng, catalog, field_size_arcsec):
    """Pick random catalog rows and random positions for one scene."""
    area_arcmin2 = (field_size_arcsec / 60.0) ** 2
    nobj = rng.poisson(area_arcmin2 * RANDOM_DENSITY)
    indices = rng.randint(0, len(catalog), size=nobj)
    half = field_size_arcsec / 2.0
    dx = rng.uniform(-half, half, size=nobj)
    dy = rng.uniform(-half, half, size=nobj)
    return indices, dx, dy


def render_scene(rng, catalog, coadd_dim, buff, bands, truth_band=TRUTH_BAND):
    """
    Render one scene: noisy images in every requested band, plus noiseless
    per-object images in `truth_band` (for computing blend truth).
    """
    field_size_arcsec = (coadd_dim - 2 * buff) * SCALE
    indices, dx, dy = sample_scene_catalog(rng, catalog, field_size_arcsec)
    rows = catalog[indices]
    nobj = len(indices)

    psf = galsim.Gaussian(fwhm=PSF_FWHM)
    cen = (coadd_dim - 1) / 2.0  # 0-indexed array-center convention

    all_bands = list(dict.fromkeys(bands + [truth_band]))
    noisy_images = {}
    per_object_truth_images = None

    for band in all_bands:
        profiles = [
            galsim.Convolve(make_galaxy_profile(rows[i], band), psf).shift(dx[i], dy[i])
            for i in range(nobj)
        ]

        if band == truth_band:
            per_object_truth_images = [
                p.drawImage(nx=coadd_dim, ny=coadd_dim, scale=SCALE).array.copy()
                for p in profiles
            ]

        if band in bands:
            full = galsim.Add(profiles)
            image = full.drawImage(nx=coadd_dim, ny=coadd_dim, scale=SCALE)
            sky_sigma = BAND_PARAMS[band]["sky_sigma"]
            image.array[:, :] += rng.normal(scale=sky_sigma, size=(coadd_dim, coadd_dim))
            noisy_images[band] = image.array.copy()

    # pixel positions in the same 0-indexed array convention as drawImage's
    # implicit centering (image center pixel = cen, offsets in arcsec/SCALE)
    positions = np.stack([cen + dx / SCALE, cen + dy / SCALE], axis=1)

    return {
        "noisy_images": noisy_images,
        "per_object_truth_images": per_object_truth_images,
        "positions": positions,
        "catalog_rows": rows,
        "coadd_dim": coadd_dim,
        "buff": buff,
        "truth_band": truth_band,
    }


def blendedness(target_img, neighbor_sum_img):
    """
    HSC/Bosch-et-al.-2018-style blendedness:
        B = 1 - <child, child> / <child, parent>
    B ~ 0 for isolated objects, -> 1 as neighbor flux dominates the footprint.
    Returns None if target has ~no flux in the window.
    """
    parent = target_img + neighbor_sum_img
    denom = np.sum(target_img * parent)
    numer = np.sum(target_img * target_img)
    if denom <= 0:
        return None
    return 1.0 - numer / denom


def extract_cutouts(scene, stamp_size, min_flux_frac_neighbor=0.01, min_detection_snr=5.0):
    """
    For every object whose stamp fits fully within the image AND that would
    plausibly be *detected* (matched-filter S/N in the truth band above
    min_detection_snr), extract the noisy multi-band cutout plus truth
    metrics (blendedness, neighbor count, nearest-neighbor separation/flux
    ratio).

    The detection cut matters: without it, ultra-faint/high-z catalog
    entries with essentially zero real flux get "classified" purely on
    floating-point noise, which can even make the blendedness ratio go
    negative (both numerator and denominator are noise-level near-zero
    numbers). Real blend-severity labels only make sense for objects that
    would actually show up as a detected source in LSST.
    """
    bands = sorted(scene["noisy_images"].keys())
    coadd_dim = scene["coadd_dim"]
    truth_imgs = scene["per_object_truth_images"]
    positions = scene["positions"]
    catalog_rows = scene["catalog_rows"]
    sky_sigma_truth = BAND_PARAMS[scene["truth_band"]]["sky_sigma"]
    nobj = len(truth_imgs)
    half = stamp_size // 2

    results = []
    for t in range(nobj):
        x, y = positions[t]
        ix, iy = int(round(x)), int(round(y))
        xlo, xhi = ix - half, ix - half + stamp_size
        ylo, yhi = iy - half, iy - half + stamp_size
        if xlo < 0 or ylo < 0 or xhi > coadd_dim or yhi > coadd_dim:
            continue

        target_img = truth_imgs[t][ylo:yhi, xlo:xhi]
        flux_target = target_img.sum()
        if flux_target <= 0:
            continue
        detection_snr = flux_target / (sky_sigma_truth * _PSF_AP_FACTOR)
        if detection_snr < min_detection_snr:
            continue  # not a plausibly detected source, skip

        neighbor_sum = np.zeros_like(target_img)
        neighbor_fluxes = []
        for i in range(nobj):
            if i == t:
                continue
            stamp_i = truth_imgs[i][ylo:yhi, xlo:xhi]
            f_i = stamp_i.sum()
            if f_i > 0:
                neighbor_sum += stamp_i
                if f_i > min_flux_frac_neighbor * flux_target:
                    dist = np.hypot(positions[i, 0] - x, positions[i, 1] - y) * SCALE
                    neighbor_fluxes.append((f_i, dist))

        b = blendedness(target_img, neighbor_sum)
        if b is None:
            continue

        n_sig_neighbors = len(neighbor_fluxes)
        if neighbor_fluxes:
            neighbor_fluxes.sort(key=lambda z: z[1])
            nearest_flux, nearest_sep = neighbor_fluxes[0]
            flux_ratio_nearest = nearest_flux / flux_target
        else:
            nearest_sep = np.inf
            flux_ratio_nearest = 0.0

        cutout = np.stack(
            [scene["noisy_images"][band][ylo:yhi, xlo:xhi] for band in bands],
            axis=0,
        )

        results.append({
            "obj_index": t,
            "x_pix": x, "y_pix": y,
            "bands": bands,
            "cutout": cutout,
            "blendedness": float(b),
            "n_sig_neighbors": n_sig_neighbors,
            "sep_nearest_arcsec": float(nearest_sep),
            "flux_ratio_nearest": float(flux_ratio_nearest),
            "redshift": float(catalog_rows["redshift"][t]),
            "r_ab": float(catalog_rows["r_ab"][t]),
        })

    return results
