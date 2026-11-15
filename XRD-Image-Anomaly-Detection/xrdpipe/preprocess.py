"""Geometry and reduction: detector mask, beam centre, azimuthal integration."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage


def detector_mask(img: np.ndarray, dead_value: float = 0.0, min_area: int = 50) -> np.ndarray:
    """True for valid pixels. Masks large connected regions at the dead value
    (module gaps, outside-detector padding) and the beamstop shadow."""
    dead = img <= dead_value
    lab, n = ndimage.label(dead)
    if n:
        sizes = ndimage.sum(dead, lab, index=np.arange(1, n + 1))
        big = np.isin(lab, np.flatnonzero(sizes >= min_area) + 1)
    else:
        big = np.zeros_like(dead)
    return ~big


@lru_cache(maxsize=8)
def _grid(shape):
    yy, xx = np.indices(shape, dtype=np.float32)
    return yy, xx


def _polar(shape, center):
    yy, xx = _grid(tuple(shape))
    dy, dx = yy - np.float32(center[0]), xx - np.float32(center[1])
    return np.hypot(dy, dx), np.arctan2(dy, dx)


def radial_profile(img, mask, center, n_bins=256, n_sectors=1):
    """Azimuthally average intensity in radial bins.

    Returns (r_centres, profile) where profile has shape (n_sectors, n_bins);
    empty bins are NaN. Sectors let us compare rings around the azimuth.
    """
    r, phi = _polar(img.shape, center)
    r_max = min(center[0], center[1], img.shape[0] - center[0], img.shape[1] - center[1])
    r_bin = np.clip((r / r_max * n_bins).astype(int), 0, n_bins)  # n_bins == outside
    s_bin = ((phi + np.pi) / (2 * np.pi) * n_sectors).astype(int) % n_sectors
    valid = mask & (r_bin < n_bins)
    idx = s_bin[valid] * n_bins + r_bin[valid]
    sums = np.bincount(idx, weights=img[valid], minlength=n_sectors * n_bins)
    cnts = np.bincount(idx, minlength=n_sectors * n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = (sums / cnts).reshape(n_sectors, n_bins)
    prof[cnts.reshape(n_sectors, n_bins) < 3] = np.nan
    r_cent = (np.arange(n_bins) + 0.5) * r_max / n_bins
    return r_cent, prof


def _symmetry_cost(img, mask, center, n_bins=64, n_sectors=8):
    """How much 8 azimuthal sectors' radial profiles disagree (lower = more symmetric)."""
    _, p = radial_profile(img, mask, center, n_bins=n_bins, n_sectors=n_sectors)
    p = np.log1p(np.nan_to_num(p, nan=np.nanmean(p)))  # log: faint outer rings count too
    return float(np.nanmean(np.nanstd(p, axis=0)))


def find_beam_center(img, mask, guess=None, search=80):
    """Find the beam centre as the most azimuthally symmetric point.

    PNGs carry no geometry header. Coarse-to-fine grid search: the symmetry
    cost is lumpy (radial binning makes it step-like), so a local optimiser
    like Nelder-Mead can stall on a plateau. Each stage searches a smaller
    window around the previous best, on a less downsampled image.
    """
    if guess is None:
        guess = (img.shape[0] / 2, img.shape[1] / 2)
    best = np.array(guess, float)
    #          (downsample, half-width px, step px)
    stages = [(8, search, 8), (4, 8, 2), (2, 2, 1)]
    for ds, half, step in stages:
        small, smask = img[::ds, ::ds], mask[::ds, ::ds]
        offs = np.arange(-half, half + 1e-9, step)
        cands = [best + [dy, dx] for dy in offs for dx in offs]
        costs = [_symmetry_cost(small, smask, c / ds) for c in cands]
        best = cands[int(np.argmin(costs))]
    return tuple(best)


def beamstop_mask(img, center, ratio=0.6, reach=60, min_area=200, downsample=2):
    """True for pixels in the beamstop shadow (disc + support arm).

    The shadow isn't zero in RefleX PNGs (often 30-60% of the local level),
    so a near-zero test misses it. Instead: flag pixels much darker than the
    median at their radius, then keep only connected regions that reach the
    beam centre, which is what a beamstop and its arm look like. Detector gaps
    and spots far from the centre are left alone.
    """
    ds = downsample
    s = ndimage.gaussian_filter(img[::ds, ::ds], 1.5)
    yy, xx = np.indices(s.shape)
    r = np.hypot(yy - center[0] / ds, xx - center[1] / ds).astype(int)
    order = np.argsort(r, axis=None)
    r_sorted, v_sorted = r.ravel()[order], s.ravel()[order]
    edges = np.flatnonzero(np.diff(r_sorted)) + 1
    meds = np.array([np.median(c) for c in np.split(v_sorted, edges)])
    radial_med = np.zeros(r.max() + 1)
    radial_med[r_sorted[np.r_[0, edges]]] = meds
    dark = s < ratio * radial_med[r]
    lab, n = ndimage.label(dark)
    if n == 0:
        return np.zeros(img.shape, bool)
    near = np.unique(lab[(r <= reach / ds) & dark])
    sizes = np.bincount(lab.ravel())
    keep = [k for k in near if k and sizes[k] * ds * ds >= min_area]
    m = ndimage.binary_dilation(np.isin(lab, keep), iterations=2)
    full = np.repeat(np.repeat(m, ds, axis=0), ds, axis=1)[: img.shape[0], : img.shape[1]]
    return full


def refine_center_and_mask(img, base_mask, n_iter=1):
    """Alternate beamstop masking and centring: a better centre gives a cleaner
    shadow mask, and a cleaner mask gives a less biased centre."""
    center = (img.shape[0] / 2, img.shape[1] / 2)
    mask = base_mask
    for _ in range(n_iter):
        mask = base_mask & ~beamstop_mask(img, center)
        center = find_beam_center(img, mask)
    mask = base_mask & ~beamstop_mask(img, center)
    return center, mask
