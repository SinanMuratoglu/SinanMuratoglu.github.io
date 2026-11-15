"""Generate fake RefleX-style images + label/split CSVs for smoke-testing

Each image gets Bragg spots, a radial background, and randomly chosen
anomalies (ice rings, background ring, strong background, diffuse humps,
non-uniform detector gradient, artifact streak, loop scatter blob)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io

CLASSES = ["Loop Scattering", "Background Ring", "Strong Background", "Diffuse Scattering",
           "Artifact", "Ice Ring", "Non-uniform Detector"]


def make(rng, n=512):
    yy, xx = np.indices((n, n))
    c = np.array([n / 2, n / 2]) + rng.normal(0, 6, 2)
    r = np.hypot(yy - c[0], xx - c[1])
    phi = np.arctan2(yy - c[0], xx - c[1])
    lab = {k: int(rng.random() < 0.3) for k in CLASSES}
    img = 30 * np.exp(-r / (n * 0.35)) + 5
    if lab["Strong Background"]: img *= 4
    if lab["Ice Ring"]:
        for rr in (0.55, 0.59, 0.63): img += 60 * np.exp(-((r - rr * n / 2) / 1.5) ** 2)
    if lab["Background Ring"]: img += 25 * np.exp(-((r - 0.3 * n / 2) / 6) ** 2)
    if lab["Diffuse Scattering"]: img += 20 * np.exp(-((r - 0.75 * n / 2) / 25) ** 2) * (1 + 0.5 * np.cos(4 * phi))
    if lab["Non-uniform Detector"]: img *= 1 + 0.6 * (xx / n)
    if lab["Loop Scattering"]: img += 80 * np.exp(-(((r - 0.15 * n / 2) / 10) ** 2 + ((phi - 1.0) / 0.3) ** 2))
    pts = rng.integers(0, n, (300, 2))
    for y, x in pts: img[max(y-1,0):y+2, max(x-1,0):x+2] += rng.uniform(50, 200)
    if lab["Artifact"]:
        row = rng.integers(0, n); img[row:row + 2] = img.max()
    img = rng.poisson(img).astype(float)
    img[r < 12] = 0  # beamstop
    img[:, n // 2 - 3 : n // 2 + 3] = 0  # module gap
    img = np.clip(img / img.max() * 255 * 1.5, 0, 255).astype(np.uint8)
    return 255 - img, lab


def main(out, count=120):
    out = Path(out); (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(count):
        img, lab = make(rng)
        name = f"img_{i:04d}.png"
        io.imsave(out / "images" / name, img, check_contrast=False)
        rows.append({"image": name, **lab})
    df = pd.DataFrame(rows)
    df.to_csv(out / "labels.csv", index=False)
    idx = rng.permutation(count)
    for k, sl in {"train": idx[: int(.7 * count)], "val": idx[int(.7 * count): int(.85 * count)],
                  "test": idx[int(.85 * count):]}.items():
        df.iloc[sl][["image"]].to_csv(out / f"labels_{k}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 120)
