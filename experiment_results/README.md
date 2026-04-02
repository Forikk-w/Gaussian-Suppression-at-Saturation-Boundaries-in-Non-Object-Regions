# Experiment Results

This directory is a curated copy of the valid experiment outputs previously stored under `~/Desktop/AIM/luminance_GS`.

## Layout

- `training_runs/`
  - Final training artifacts for the 8 scene-condition runs kept for direct inspection.
  - Renamed for consistency:
    - `chair_over_exp_rerun -> chair_over_exp`
    - `sofa_low_rerun -> sofa_low`
    - `sofa_over_exp_rerun -> sofa_over_exp`
- `analysis/`
  - `exp01_opacity_scale/`
  - `exp07_importance_pruning/`
  - `exp09_importance_pruning_verified/`
  - `exp10_cap_to_low/`
  - `exp11_importance_distribution_verified/`
  - `exp12_density_comparison_raw_count/`
  - `exp13_densification_event_maps/`

## Curation Rules

- Old outputs replaced by newer corrected versions were omitted.
- Desktop cache files such as `.DS_Store` were removed.
- `exp08_pruning_curve` was omitted because the verified replacement is `analysis/exp09_importance_pruning_verified/pruning_curve/`.
- `exp12_density_comparison` was omitted because the corrected raw-count version is `analysis/exp12_density_comparison_raw_count/`.
- `exp13` keeps the raw `densification_event_maps/` plus the final boundary/interior/normal analysis only.
  - Renamed from the desktop `v2` outputs:
    - `densification_analysis_v2.csv -> densification_analysis.csv`
    - `densification_report_v2.json -> densification_report.json`
    - `*_densification_compare_v2.png -> *_densification_compare.png`
    - `all_scenes_overview_v2.png -> all_scenes_overview.png`

## Code Pairing

The corresponding final analysis/training code is kept under [`Luminance-GS/examples`](/Users/4v1/PycharmProjects/Luminance-GS/Luminance-GS/examples).

- The old `exp13` analysis entry was replaced by the final version in
  [`analyze_exp13_densification_event_maps.py`](/Users/4v1/PycharmProjects/Luminance-GS/Luminance-GS/examples/analyze_exp13_densification_event_maps.py).
- The superseded `analyze_exp8_pruning_curve.py` script was removed.
