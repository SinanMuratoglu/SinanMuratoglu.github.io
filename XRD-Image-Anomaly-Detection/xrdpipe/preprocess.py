"""Geometry and reduction: detector mask, beam centre, azimuthal integration."""
from __future__ import annotations

import numpy as np
from scipy import ndimage, optimize


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


def _polar(shape, center):
    yy, xx = np.indices(shape)
    dy, dx = yy - center[0], xx - center[1]
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


def find_beam_center(img, mask, guess=None, search=40, downsample=4):
    """Refine the beam centre by maximising azimuthal symmetry.

    PNGs carry no geometry header, so we pick the centre that minimises the
    spread between 8 azimuthal sectors of the radial profile. Done on a
    downsampled image for speed, then scaled back.
    """
    if guess is None:
        guess = (img.shape[0] / 2, img.shape[1] / 2)
    small = img[::downsample, ::downsample]
    smask = mask[::downsample, ::downsample]
    g = np.array(guess) / downsample

    def cost(c):
        _, p = radial_profile(small, smask, c, n_bins=64, n_sectors=8)
        p = np.log1p(np.nan_to_num(p, nan=np.nanmean(p)))
        return float(np.nanmean(np.nanstd(p, axis=0)))

    res = optimize.minimize(cost, g, method="Nelder-Mead",
                            options={"xatol": 0.25, "fatol": 1e-5, "initial_simplex":
                                     [g, g + [search / downsample / 4, 0], g + [0, search / downsample / 4]]})
    c = res.x * downsample
    # keep the refined centre only if it stayed within the search window
    if np.hypot(*(c - np.array(guess))) > search:
        return tuple(guess)
    return tuple(c)
