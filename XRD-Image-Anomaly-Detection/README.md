# xrdpipe — anomaly detection in X-ray diffraction images

A Python pipeline that takes raw diffraction images from the RefleX dataset
(Czyzewski et al., Zenodo 10.5281/zenodo.2605120) and flags seven kinds of
image problems: ice rings, background rings, strong background, diffuse
scattering, loop scattering, non-uniform detector response, and artifacts.

gradient-boosted classifier runs on interpretable features:

1. **Detector mask**: drops module gaps and the beamstop shadow.
2. **Beam center refinement**: PNGs have no geometry header so center is
   found by maximizing azimuthal symmetry across 8 sectors
3. **Azimuthal integration**: 256 radial bins × 16 azimuthal sectors. 1D profile
   is median across sectors
4. **Features**: sharp rings (with how many sectors each ring lights up),
   background level/slope, broad diffuse humps, sector asymmetry (Fourier
   harmonics), Bragg spot density (white top-hat), saturation/dead-area/streak
   scores. About 30 features per image.
5. **Model**: one HistGradientBoosting classifier per class, with per-class thresholds
   tuned on the validation split. The report covers ROC-AUC, average precision and F1
   on the held-out test split, plus permutation feature importance.

**Project page:** https://sinanmuratoglu.github.io/XRD-Image-Anomaly-Detection/

## Usage

```bash
pip install -r requirements.txt
# one image -> diagnostic figure + feature printout
python -m xrdpipe inspect path/to/image.png
# whole dataset -> features.csv (resumable; safe to stop and rerun)
python -m xrdpipe extract --images reflex_img_1024_inter_nearest/ --labels labels.csv --out features.csv --jobs 8
# train + evaluate on the official splits
python -m xrdpipe train --features features.csv --labels labels.csv \
    --train labels_train.csv --val labels_val.csv --test labels_test.csv --out results/
```

`tests/make_synthetic.py OUT_DIR N` generates fake labelled images for smoke tests

`scripts/make_hero.py IMAGE OUT.png` and `scripts/make_heatmap.py results/feature_importance.csv OUT.png`
regenerate the figures on the project page (run with `PYTHONPATH=.`).
`scripts/compare_baselines.py` + `scripts/make_comparison.py` reproduce the baseline comparison chart

The images aren't in this repo (3.6 GB). Download them from Zenodo:
[10.5281/zenodo.2605120](https://doi.org/10.5281/zenodo.2605120) (CC BY 4.0, Czyzewski et al., 2019)

