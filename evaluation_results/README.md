# Evaluation Benchmarks & Metrics — Lunar PSR Terminal

This directory archives verified performance evaluation metrics, threshold sweeps, hazard engine benchmarks, and visual error analysis artifacts.

## Contents
- `metrics.json` & `metrics.csv`: Primary restoration benchmark metrics across held-out evaluation patches.
- `lunar_unet_complete_metrics.json` & `.csv`: Comprehensive multi-metric breakdown including MAE, RMSE, PSNR, SSIM, and R².
- `lunar_unet_metric_methodology.md`: Complete methodology explaining metric derivations and test setup.
- `hazard_v2/`:
  - `best_validation_configs.json`: Optimal hazard engine parameters for `balanced`, `conservative`, and `precision` operating modes.
  - `precision_recall_curve.csv` & `pareto_frontier.csv`: PR frontier for hazard detection.
  - `threshold_sweep.csv`: Multidimensional grid search across detection thresholds and component area filters.
- `v2_final/`:
  - `v2_final_comparison.json`: Side-by-side comparison of Baseline vs. V1 Improved vs. V2 Multiscale hazard engine.
  - `visual_analysis/`: High-resolution visual comparisons highlighting crater rims, boulder fields, and steep slope hazards.
- `final/`:
  - Historical comparison records and error analysis samples.
