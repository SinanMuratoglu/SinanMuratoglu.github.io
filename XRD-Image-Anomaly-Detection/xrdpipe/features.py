"""Physics-motivated features for each RefleX anomaly class.

Rough mapping of feature groups to the classes they target:
  ring_*      sharp, azimuthally uniform rings  -> Ice Ring, Background Ring
  bg_*        overall level / slope of the radial background -> Strong Background
  diffuse_*   broad humps in the profile, low-frequency texture -> Diffuse Scattering
  sector_*    azimuthal asymmetry                -> Non-uniform Detector, Loop Scattering
  spot_*      Bragg spot density (context: good crystal vs. none)
  art_*       saturation, dead pixels, streaks   -> Artifact
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage, signal
from skimage.morphology import disk, white_tophat

from .preprocess import detector_mask, radial_profile, refine_center_and_mask

N_BINS = 256
N_SECTORS = 16
R_MIN_PX = 32  # beamstop + its shadow edge


def _fill_nan(x):
    x = x.copy()
    bad = ~np.isfinite(x)
    if bad.all():
        return np.zeros_like(x)
    x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad])
    return x


def extract_features(img: np.ndarray, refine_center: bool = True) -> tuple[dict, dict]:
    """Return (features, diagnostics). Diagnostics hold arrays for plotting."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _extract(img, refine_center)


def _extract(img, refine_center):
    mask = detector_mask(img, dead_value=6.0)
    if refine_center:
        center, mask = refine_center_and_mask(img, mask)
    else:
        center = (img.shape[0] / 2, img.shape[1] / 2)
    # median over azimuthal sectors: robust to Bragg spots, keeps true rings
    r, sect = radial_profile(img, mask, center, N_BINS, N_SECTORS)
    with np.errstate(all="ignore"):
        prof = _fill_nan(np.nanmedian(sect, axis=0))
    lp = np.log1p(prof)

    f: dict[str, float] = {}
    f["center_dy"] = center[0] - img.shape[0] / 2
    f["center_dx"] = center[1] - img.shape[1] / 2
    vals = img[mask]
    f["int_mean"], f["int_std"] = float(vals.mean()), float(vals.std())
    f["int_p50"], f["int_p99"] = np.percentile(vals, [50, 99]).tolist()

    # --- background: smooth baseline via wide median filter on the log profile
    base = ndimage.median_filter(lp, size=N_BINS // 3, mode="nearest")
    resid = lp - base
    f["bg_level"] = float(base.mean())
    f["bg_inner"] = float(base[: N_BINS // 4].mean())
    f["bg_outer"] = float(base[-N_BINS // 4 :].mean())
    f["bg_slope"] = float(np.polyfit(r, base, 1)[0] * r.max())

    # --- rings: narrow positive residual peaks
    noise = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-6
    # skip the beamstop region: its edge is a step, not a ring
    r_min = int(np.searchsorted(r, R_MIN_PX))
    peaks, props = signal.find_peaks(resid[r_min:], prominence=4 * noise, width=(1, 12))
    peaks = peaks + r_min
    f["ring_count"] = len(peaks)
    prom = props["prominences"] if len(peaks) else np.array([0.0])
    f["ring_max_prom"] = float(prom.max() / noise)
    f["ring_sum_prom"] = float(prom.sum() / noise)
    # azimuthal uniformity of the strongest ring: a real ring is lit in every sector
    # 2nd and 3rd sharpest rings (the sharpest is ring_max_prom above)
    ranked = np.sort(prom)[::-1] / noise
    for i in (1, 2):
        f[f"ring_prom_{i + 1}"] = float(ranked[i]) if i < len(ranked) else 0.0
    if len(peaks):
        k = peaks[np.argmax(prom)]
        ring_sect = sect[:, max(k - 1, 0) : k + 2].mean(axis=1)
        local_bg = np.nanmean(sect[:, max(k - 8, 0) : max(k - 4, 1)], axis=1)
        contrast = ring_sect - local_bg
        f["ring_frac_sectors_lit"] = float(np.nanmean(contrast > 0))
        f["ring_radius_frac"] = float(k / N_BINS)
    else:
        f["ring_frac_sectors_lit"] = 0.0
        f["ring_radius_frac"] = 0.0

    # --- diffuse: broad structure = medium-scale smoothing minus wide baseline
    mid = ndimage.gaussian_filter1d(lp, 6) - base
    f["diffuse_energy"] = float(np.sqrt(np.mean(mid**2)))
    bpk, bprops = signal.find_peaks(mid, prominence=0.05, width=10)
    f["diffuse_humps"] = len(bpk)

    # --- sector asymmetry
    ls = np.log1p(sect)
    sector_mean = np.nanmean(ls, axis=1)
    f["sector_cv"] = float(np.nanstd(sector_mean) / (abs(np.nanmean(sector_mean)) + 1e-6))
    f["sector_range"] = float(np.nanmax(sector_mean) - np.nanmin(sector_mean))
    f["sector_spread_by_r"] = float(np.nanmean(np.nanstd(ls, axis=0)))
    # first harmonic (one side brighter) vs second (quadrupole) of sector means
    sm = _fill_nan(sector_mean) - np.nanmean(sector_mean)
    fft = np.abs(np.fft.rfft(sm))
    f["sector_h1"], f["sector_h2"] = float(fft[1]), float(fft[2])

    # --- Bragg spots: top-hat removes smooth background, count bright blobs
    small = img[::2, ::2]
    th = white_tophat(np.log1p(small), disk(3))
    thr = th.mean() + 5 * th.std()
    lab, n_spots = ndimage.label(th > thr)
    f["spot_count"] = int(n_spots)
    f["spot_frac"] = float((th > thr).mean())

    # --- artifacts: saturation, dead area, straight streaks
    vmax = img.max()
    f["art_saturated_frac"] = float((img >= vmax).mean()) if vmax > 0 else 0.0
    f["art_masked_frac"] = float(1 - mask.mean())
    row_prof = np.log1p(small).mean(axis=1)
    col_prof = np.log1p(small).mean(axis=0)
    hp = lambda x: x - ndimage.median_filter(x, 15, mode="nearest")
    f["art_line_score"] = float(max(np.abs(hp(row_prof)).max(), np.abs(hp(col_prof)).max()))

    diag = {"r": r, "profile": prof, "baseline": np.expm1(base), "peaks": peaks,
            "center": center, "mask": mask, "sectors": sect}
    return f, diag
