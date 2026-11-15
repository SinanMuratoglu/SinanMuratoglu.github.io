---
title: Flagging Bad X-ray Diffraction Images
---

![An ice-contaminated diffraction image and its ring profile](figs/hero_ice_ring.png)

<img src="figs/baseline_comparison.png" alt="" style="width: 600px; height: auto;">

# Flagging Bad X-ray Diffraction Images

Pipeline that flags raw X-ray diffraction images for seven kinds of problems: ice rings, background rings, strong background, diffuse scattering, loop scattering, non-uniform detector response, and artifacts. 

## Workflow
Instead of training a black-box neural network on the pixels, it reduces each image the way a person might read it, so the model can show the reason for flagging an image.

1. **Mask the detector.** Ignore the beamstop, its shadow, and gaps between detector modules.
2. **Find the beam center.** The images come with no geometry information, so I pick the center that makes the pattern most symmetric around it.
3. **Average intensity in rings.** I bin intensity by distance from the center (256 bins), split into 16 angular sectors. Taking the median across sectors keeps single Bragg spots from looking like rings.
4. **Extract features.** 29 numbers per image, each aimed at a problem class: how sharp the strongest ring is and whether it goes all the way around, background level and slope, broad diffuse humps, whether one side of the detector is brighter than the other, Bragg spot density, and saturation, dead areas, and straight streaks.
5. **Classify.** One gradient-boosted classifier per problem class (scikit-learn), with decision thresholds tuned on a validation set.

## Results

On the dataset's held-out test set:

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

ROC-AUC (0-0.1) measures how well the model ranks images with a problem above images without it. F1 measures how accurate the yes/no choices are.

![Heatmap of which features the model relies on for each problem](figs/feature_heatmap.png)

To find what the model relies on, I shuffled one feature at a time on the validation set and measured how much the score dropped ([permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)). For four of the problems, the reasoning lines up with the physics:

- **Ice rings** are flagged mainly by whether the sharpest ring goes all the way around and where it sits. 
- **Background rings** are flagged by broad humps in the ring profile.
- **Strong background** comes almost entirely from the background level.
- **Artifacts** are flagged by the straight-streak detector.

The other three are less convincing. Loop scattering, non-uniform detector response and diffuse scattering lean on the fraction of saturated pixels. That's a correlation in the dataset, not the physical signature of those problems, and it's where the scores are lowest. Diffuse scattering also has only 44 examples in the test set, so its yes/no calls are unreliable even though the ranking is decent. Features that describe the shape of those signals directly are the obvious next step.

## Bugs & Fixes

- **The images were stored inverted.** I wrote the pipeline initially thinking that spots from scattering would be bright, but the real data is the opposite; but after running the program on a smaller set saw something was definitely wrong; instead of changing the entire pipeline, I just invert each image before it is analyzed.
- **Image IDs with dots were being cut off.** Some IDs contain a period, and the standard "strip the file extension" call chopped them at that period. That dropped about 900 labeled images from training until I counted matches and added them back
- **The beamstop edge looked like a ring.** The sharp step where the beamstop shadow ends gave the strongest "ring" for some images, so the ring search skips the inner 32 pixels.

## Code

Written in Python with NumPy, SciPy, scikit-image, and scikit-learn. The code is in this folder: [`xrdpipe/`](xrdpipe/) has the pipeline; see [`README.md`](README.md) for instructions.

*Dataset: A. Czyzewski, F. Krawiec, D. Brzezinski, P. J. Porebski, "RefleX: X-ray diffraction images dataset," Zenodo (2019), [10.5281/zenodo.2605120](https://doi.org/10.5281/zenodo.2605120), CC BY 4.0.*

[← Back to projects](../)
