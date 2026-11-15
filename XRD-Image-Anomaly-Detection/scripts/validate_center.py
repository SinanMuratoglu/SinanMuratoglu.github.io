"""Validate beam-centre finding against circle fits to ice rings.

Ice rings are sharp, complete circles centred on the beam, so fitting a circle
to one gives an independent estimate of the true centre. Compares the old
centres (stored in data/features.csv) with the current code.
Usage (from repo root): python3 scripts/validate_center.py [N]
"""
import sys, random
import numpy as np, pandas as pd
from scipy import ndimage, optimize
from xrdpipe.io import load_image, find_images, load_labels
from xrdpipe.preprocess import detector_mask, refine_center_and_mask, radial_profile


def ring_fit(a, guess, r_lo, r_hi):
    s = ndimage.gaussian_filter(a, 1.5); pts = []
    for t in np.linspace(0, 2 * np.pi, 180, endpoint=False):
        rr = np.arange(r_lo, r_hi, 0.5); y = guess[0] + rr * np.sin(t); x = guess[1] + rr * np.cos(t)
        ok = (y > 0) & (y < a.shape[0] - 1) & (x > 0) & (x < a.shape[1] - 1)
        if ok.sum() < 10: continue
        v = ndimage.map_coordinates(s, [y[ok], x[ok]], order=1); k = np.argmax(v)
        if v[k] > np.median(v) * 1.15: pts.append((y[ok][k], x[ok][k]))
    P = np.array(pts)
    if len(P) < 120: return None
    f = lambda c: np.std(np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]))
    c = optimize.minimize(f, guess, method='Nelder-Mead').x
    spread = f(c)
    return c if spread < 2.0 else None     # only trust clean, complete rings


N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
im = find_images('images/reflex_img_1024_inter_nearest')
L = load_labels('data/labels.csv'); F = pd.read_csv('data/features.csv', index_col=0); F.index = F.index.astype(str)
ice = sorted(L.index[L['Ice Ring'] == 1]); random.seed(0); random.shuffle(ice)
rows = []
for n in ice:
    if len(rows) >= N: break
    a = load_image(im[n]); c_new, m = refine_center_and_mask(a, detector_mask(a, dead_value=6.0))
    r, p = radial_profile(a, m, c_new, 256, 1); p = np.nan_to_num(p[0])
    base = ndimage.median_filter(np.log1p(p), 85); k = 16 + int(np.argmax((np.log1p(p) - base)[16:]))
    truth = ring_fit(a, c_new, r[k] - 20, r[k] + 20)
    if truth is None: continue
    c_old = np.array([512 + F.loc[n, 'center_dy'], 512 + F.loc[n, 'center_dx']])
    rows.append((n, np.hypot(*(c_old - truth)), np.hypot(*(np.array(c_new) - truth))))
R = pd.DataFrame(rows, columns=['image', 'old_err_px', 'new_err_px'])
print(R.round(2).to_string(index=False))
print(f"\n{len(R)} ice images with a clean ring fit")
for col in ('old_err_px', 'new_err_px'):
    e = R[col]; print(f"{col}: median {e.median():.2f}  90th pct {e.quantile(.9):.2f}  max {e.max():.2f}  >5px {np.mean(e > 5):.0%}")
