"""Loading RefleX images and label files.

The RefleX label CSVs aren't documented on Zenodo, so the loader detects
the layout instead of assuming one. Two layouts are handled:
  * wide:  image, Ice ring, Diffuse Scattering, ...  (0/1 per class)
  * long:  image, labels   where labels is a delimited string of class names
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io as skio


def _stem(s) -> str:
    """Strip a .png extension only; RefleX ids can contain dots."""
    s = Path(str(s)).name
    return s[:-4] if s.lower().endswith(".png") else s

CLASSES = [
    "Loop Scattering",
    "Background Ring",
    "Strong Background",
    "Diffuse Scattering",
    "Artifact",
    "Ice Ring",
    "Non-uniform Detector",
]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower())


_CANON = {_norm(c): c for c in CLASSES}


def load_labels(csv_path: str | Path) -> pd.DataFrame:
    """Return a DataFrame indexed by image stem with one 0/1 column per class."""
    df = pd.read_csv(csv_path)
    id_col = df.columns[0]
    ids = df[id_col].astype(str).map(_stem)

    class_cols = {c: _CANON[_norm(c)] for c in df.columns[1:] if _norm(c) in _CANON}
    if class_cols:  # wide layout
        out = df[list(class_cols)].rename(columns=class_cols).fillna(0).astype(int)
    else:  # long layout: find the first column holding class names
        text_col = next(
            (c for c in df.columns[1:] if df[c].astype(str).map(_norm).str.contains("|".join(_CANON)).any()),
            None,
        )
        if text_col is None:
            # split files may only list image ids
            return pd.DataFrame(index=pd.Index(ids, name="image"))
        out = pd.DataFrame(0, index=range(len(df)), columns=CLASSES)
        for i, cell in df[text_col].astype(str).items():
            for tok in re.split(r"[;,|]", cell):
                key = _norm(tok)
                if key in _CANON:
                    out.loc[i, _CANON[key]] = 1
    out.index = pd.Index(ids, name="image")
    for c in CLASSES:
        if c not in out:
            out[c] = 0
    return out[CLASSES]


def load_split_ids(csv_path: str | Path) -> list[str]:
    df = pd.read_csv(csv_path)
    return df[df.columns[0]].astype(str).map(_stem).tolist()


def find_images(image_dir: str | Path) -> dict[str, Path]:
    """Map image stem -> path for every PNG under image_dir (recursive)."""
    return {p.stem: p for p in Path(image_dir).rglob("*.png") if not p.name.startswith("._")}


def load_image(path: str | Path, invert: bool = True) -> np.ndarray:
    """Load a diffraction PNG as float32, larger = more photons."""
    img = skio.imread(path)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)
    img = img.astype(np.float32)
    # RefleX PNGs use the crystallographer display convention (dark = strong
    # scattering, beamstop = 255); all 150 sampled images had median > 199.
    # Invert so larger = more photons; beamstop -> 0 (masked), saturation -> 255.
    return 255.0 - img if invert else img
