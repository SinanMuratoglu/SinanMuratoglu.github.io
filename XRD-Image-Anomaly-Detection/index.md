---
title: Flagging Bad X-ray Diffraction Images
---

![An ice-contaminated diffraction image and its ring profile](figs/hero_ice_ring.png)

# Flagging Bad X-ray Diffraction Images

A Python pipeline that looks at raw X-ray diffraction images and flags seven kinds of problems: ice rings, background rings, strong background, diffuse scattering, loop scattering, non-uniform detector response, and artifacts. Instead of training a black-box neural network on the pixels, it reduces each image the way a crystallographer would read it, so the model can show *why* it flagged an image.

## The problem

In X-ray crystallography, a crystal is rotated in the beam while a detector records thousands of diffraction images. Problems like ice forming on the sample or the beam clipping the sample loop can quietly ruin a dataset, and the experiment may not be repeatable. Nobody can check every frame by hand, so it helps to have something that scans every image and raises a flag.

I used the [RefleX dataset](https://doi.org/10.5281/zenodo.2605120) (Czyzewski et al.): 6,311 real diffraction images, each labeled by experts with any combination of the seven problem classes.

## How it works

1. **Mask the detector.** Ignore the beamstop, its shadow, and gaps between detector modules.
2. **Find the beam center.** The images come with no geometry information, so I pick the center that makes the pattern most symmetric around it.
3. **Average intensity in rings.** I bin intensity by distance from the center (256 bins), split into 16 angular sectors. Taking the median across sectors keeps single bright Bragg spots from looking like rings.
4. **Extract features.** About 30 numbers per image, each aimed at a problem class: how sharp the strongest ring is and whether it goes all the way around, background level and slope, broad diffuse humps, whether one side of the detector is brighter than the other, Bragg spot density, and saturation, dead areas, and straight streaks.
5. **Classify.** One gradient-boosted classifier per problem class (scikit-learn), with decision thresholds tuned on a validation set.

## Results

On the dataset's held-out test set (632 images, never used for training or tuning):

| Problem | ROC-AUC | F1 |
|---|---|---|
| Strong background | 0.983 | 0.93 |
| Ice ring | 0.972 | 0.78 |
| Artifact | 0.931 | 0.77 |
| Background ring | 0.924 | 0.89 |
| Diffuse scattering | 0.918 | 0.44 |
| Non-uniform detector | 0.895 | 0.77 |
| Loop scattering | 0.862 | 0.74 |
| **Average** | **0.927** | **0.76** |

ROC-AUC measures how well the model ranks images with a problem above images without it (1.0 is perfect). F1 measures how accurate its yes/no calls are.

![Heatmap of which features the model relies on for each problem](figs/feature_heatmap.png)

To see what the model actually relies on, I shuffled one feature at a time on the validation set and measured how much the score dropped (permutation importance). For four of the problems, the reasoning lines up with the physics:

- **Ice rings** are flagged mainly by whether the sharpest ring goes all the way around, plus where it sits. Ice forms a powder, which makes complete rings at fixed radii, unlike the crystal's discrete spots.
- **Background rings** are flagged by broad humps in the ring profile. These are the wide, soft rings that come from the solvent and the sample holder.
- **Strong background** comes almost entirely from the background level (83%).
- **Artifacts** are flagged by the straight-streak detector.

The other three are less convincing. Loop scattering, non-uniform detector response and diffuse scattering lean on the fraction of saturated pixels. That's a correlation in the dataset, not the physical signature of those problems, and it's where the scores are lowest. Diffuse scattering also has only 44 examples in the test set, so its yes/no calls are unreliable even though the ranking is decent. Features that describe the shape of those signals directly are the obvious next step.

## Is it actually working?

Plain accuracy would be misleading here. Only 46 of the 632 test images have ice, so a model that always answers "no ice" would be 93% accurate while catching nothing. Instead, I compared the pipeline against two baselines on the same test images:

- **Random guessing.** Its average precision is just the fraction of images that have the problem.
- **Brightness stats only.** The same kind of classifier, but trained on just four numbers per image: mean, spread, median, and brightest-1% intensity. This tests whether the physics features matter, or whether "bright image vs. dark image" is enough.

![Average precision of the full pipeline vs. two baselines for each problem](figs/baseline_comparison.png)

The full pipeline beats both baselines on every problem: 0.83 average precision on average, against 0.60 for brightness stats and 0.31 for random guessing. The biggest jump is ice rings, from 0.33 to 0.89. Brightness can't tell you whether a sharp ring exists, so the ring features are doing the work. Strong background barely improves (0.96 to 0.98), which makes sense because brightness basically defines that problem.

In practical terms, the model catches 80% of ice-ring images, and 80% of the images it flags actually have ice. Pushing it to catch 91% means only about half of its flags are real. That's the usual tradeoff between catching more problems and raising more false alarms.

## Bugs that taught me the most

- **The images were stored inverted.** RefleX saves images the way crystallographers view them: dark means strong scattering, and the beamstop is white. My first version assumed bright meant intense, so it was effectively analyzing photographic negatives. I checked 150 random images, confirmed they all used the same convention, and inverted them on load.
- **Image IDs with dots were being cut off.** Some IDs contain a period, and the standard "strip the file extension" call chopped them at that period. That silently dropped about 900 labeled images from training until I counted matches and noticed the gap.
- **The beamstop edge looked like a ring.** The sharp step where the beamstop shadow ends gave the strongest "ring" in some images, so the ring search now skips the inner 32 pixels.

## Code

Written in Python with NumPy, SciPy, scikit-image, and scikit-learn. The code is in this folder: [`xrdpipe/`](xrdpipe/) has the pipeline, and [`README.md`](README.md) explains how to run it.

*Dataset: A. Czyzewski, F. Krawiec, D. Brzezinski, P. J. Porebski, "RefleX: X-ray diffraction images dataset," Zenodo (2019), [10.5281/zenodo.2605120](https://doi.org/10.5281/zenodo.2605120), CC BY 4.0.*

[← Back to projects](../)
