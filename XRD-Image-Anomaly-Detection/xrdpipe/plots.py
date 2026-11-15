"""Diagnostic figure for a single image."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def diagnostic_figure(img, diag, feats, title="", path=None):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1.4, 1]})
    disp = np.log1p(np.where(diag["mask"], img, 0))
    ax[0].imshow(disp, cmap="gray_r", vmin=np.percentile(disp, 1), vmax=np.percentile(disp, 99.7))
    cy, cx = diag["center"]
    ax[0].plot(cx, cy, "r+", ms=14, mew=2)
    rmax = diag["r"].max()
    for k in diag["peaks"]:
        ax[0].add_patch(plt.Circle((cx, cy), diag["r"][k], fill=False, ec="tab:orange", lw=0.8, alpha=0.7))
    ax[0].set_title(title or "image (log)")
    ax[0].axis("off")

    r, p = diag["r"], diag["profile"]
    ax[1].semilogy(r, p, lw=1, label="radial profile")
    ax[1].semilogy(r, diag["baseline"], "--", lw=1, label="baseline")
    ax[1].semilogy(r[diag["peaks"]], p[diag["peaks"]], "v", c="tab:orange", label="rings")
    ax[1].set_xlabel("radius (px)")
    ax[1].set_ylabel("mean intensity")
    ax[1].set_xlim(0, rmax)
    ax[1].legend()
    ax[1].set_title("azimuthal integration")

    s = np.log1p(diag["sectors"])
    ax[2].imshow(s, aspect="auto", cmap="magma", extent=[0, rmax, 360, 0])
    ax[2].set_xlabel("radius (px)")
    ax[2].set_ylabel("azimuth (deg)")
    ax[2].set_title("sector profiles (log)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig
