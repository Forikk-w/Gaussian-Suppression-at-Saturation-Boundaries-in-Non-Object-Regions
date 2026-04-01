# Gaussian Suppression at Saturation Boundaries in Non-Object Regions

A research project investigating and suppressing redundant Gaussian proliferation at overexposed saturation boundaries in 3D Gaussian Splatting, built on top of [Luminance-GS](https://github.com/cuiziteng/Luminance-GS) (CVPR 2025).

---

## Motivation

When training 3DGS under overexposed conditions, we observe a consistent and systematic phenomenon: **the number of Gaussians generated under overexposure is approximately 2× that of low-light conditions across all 5 scenes of the LOM dataset**, yet this does not lead to proportional improvements in rendering quality (PSNR/SSIM/LPIPS).

Through a series of experiments, we identify the following:

- Overexposed images exhibit hard pixel clipping at saturation boundaries (pixel value ≈ 255), creating steep spatial gradients that persistently trigger Gaussian densification (split/clone).
- The resulting redundant Gaussians are **small in scale but high in opacity** — they are not transparent "ghost" Gaussians, but dense, compact Gaussians fitting a non-existent structure induced by sensor saturation.
- In geometrically simple scenes (sofa, buu, bike), these Gaussians are spatially concentrated at saturation boundaries with enrichment ratios up to **11.84×** above random expectation.
- Critically, saturation boundaries can be decomposed into two types: **background saturation** (caused by sensor limits on uniform bright regions) and **object edge saturation** (coinciding with real geometric boundaries). Only the former produces truly redundant Gaussians that should be suppressed.

---

## Research Questions

1. Do redundant Gaussians at saturation boundaries differ in scale/opacity between background and object-edge regions?
2. Can we design a densification suppression strategy targeting background saturation regions without harming object boundary reconstruction?
3. Does suppression reduce Gaussian count while maintaining or improving rendering quality across PSNR, SSIM, and LPIPS?

---

## Approach

This project explores two complementary directions:

**Direction 1: Training-time Densification Suppression**
Modify the adaptive density control (ADC) mechanism in Luminance-GS to detect background saturation regions during training and reduce the densification weight for Gaussians projected onto these regions. This prevents the proliferation of Gaussians driven by saturation-induced pseudo-gradients.

**Direction 2: Post-training Pruning**
After training, identify and remove Gaussians that are both (1) spatially concentrated at background saturation boundaries and (2) contribute minimally to rendering quality. This serves as a lightweight alternative to training-time modification and provides an interpretable analysis of redundancy.

Both directions rely on distinguishing **background saturation** from **object-edge saturation**, using depth gradient maps or semantic segmentation (SAM) to separate the two.

---

## Experimental Setup

- **Baseline**: [Luminance-GS](https://github.com/cuiziteng/Luminance-GS) (CVPR 2025)
- **Dataset**: [LOM Dataset](https://github.com/cuiziteng/Aleth-NeRF) — 5 scenes (buu, chair, sofa, bike, shrub) under low-light and overexposure conditions
- **Metrics**: PSNR ↑, SSIM ↑, LPIPS ↓, num_gaussians ↓

---

## Preliminary Findings

### Experiment 1: Gaussian Attribute Distribution

| Scene | Low-light tiny_scale ratio | Overexp tiny_scale ratio | Low-light opacity mean | Overexp opacity mean |
|-------|--------------------------|------------------------|----------------------|---------------------|
| bike  | 0.060 | 0.066 | 0.198 | 0.272 |
| buu   | 0.023 | 0.052 | 0.192 | 0.284 |
| chair | 0.004 | 0.010 | 0.151 | 0.256 |
| shrub | 0.068 | 0.142 | 0.188 | 0.216 |
| sofa  | 0.043 | 0.130 | 0.202 | 0.308 |

Overexposure consistently produces more tiny-scale, higher-opacity Gaussians — "small but solid" rather than transparent ghost Gaussians.

### Experiment 2: Pruning Analysis

Pruning 30% of low-opacity Gaussians causes less than 0.15 dB PSNR drop across all scenes, confirming that a substantial fraction of Gaussians contribute minimally to rendering. Notably, pruning 50% of Gaussians in `bike_low` actually **improves** PSNR by +0.47 dB, indicating overfitting in low-light conditions.

### Experiment 3: Spatial Enrichment at Saturation Boundaries

| Scene | Boundary area ratio | Tiny GS on boundary ratio | Enrichment |
|-------|--------------------|-----------------------------|------------|
| bike  | 16.7% | 51.2% | **3.06×** |
| buu   | 19.0% | 83.2% | **4.39×** |
| chair | 21.9% | 9.3%  | 0.42× |
| shrub | 12.1% | 11.3% | 0.94× |
| sofa  | 2.1%  | 24.2% | **11.84×** |

The saturation boundary hypothesis holds in geometrically simple scenes (bike, buu, sofa) but not in high-texture scenes (shrub), where scene-intrinsic complexity dominates.

---

## Project Status

- [x] Baseline reproduction (Luminance-GS on LOM dataset)
- [x] Experiment 1: Gaussian attribute distribution analysis
- [x] Experiment 2: Pruning experiment
- [x] Experiment 3: Spatial enrichment analysis
- [ ] Experiment 4: Background vs. object-edge saturation boundary decomposition
- [ ] Training-time densification suppression implementation
- [ ] Post-training pruning pipeline
- [ ] Full evaluation and comparison

---

## Related Work

- **Luminance-GS** (CVPR 2025): Adapting 3DGS to challenging lighting via view-adaptive curve adjustment. [[paper]](https://arxiv.org/abs/2504.01503) [[code]](https://github.com/cuiziteng/Luminance-GS)
- **Luminance-GS++** (arXiv 2026): Extended version with local pixel-wise residual refinement. [[paper]](https://arxiv.org/abs/2602.18322)
- **Revising Densification in Gaussian Splatting** (2024): Pixel-error driven density control. [[paper]](https://arxiv.org/abs/2404.06109)
- **GDAGS** (2025): Gradient-direction-aware density control to suppress over-densification. [[paper]](https://arxiv.org/abs/2508.09239)
- **Improving Densification in 3DGS** (2025): Growth control and recovery-aware pruning. [[paper]](https://arxiv.org/abs/2508.12313)

---

## Acknowledgements

This project builds on [Luminance-GS](https://github.com/cuiziteng/Luminance-GS). We thank the authors for their open-source contribution.
