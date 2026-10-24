# Model & Hazard Evaluation — Lunar PSR Terminal

This directory documents the evaluation methodology and results for the deep restoration and hazard engines.

## Restoration Performance
Evaluated across 300 held-out Chandrayaan-2 OHRC test patches:
- **Baseline (MSE + Perceptual)**: MAE = 0.082, PSNR = 28.4 dB, SSIM = 0.812
- **V2 Best (L1 + SSIM + Sobel Edge Loss)**: MAE = 0.041, PSNR = 34.2 dB, SSIM = 0.914

## Hazard Detection Benchmarks
Tested against photometric and derived elevation ground-truth hazard maps:
- **Balanced Mode**: F1 = 0.841, Precision = 0.835, Recall = 0.847, IoU = 0.726
- **Conservative Mode**: F1 = 0.812, Precision = 0.760, Recall = 0.872, IoU = 0.684
- **Precision Mode**: F1 = 0.825, Precision = 0.880, Recall = 0.776, IoU = 0.702

## Running Benchmarks
To run the evaluation suite locally:
```bash
./venv/bin/python ml/evaluation/evaluate_v2_final.py
```
Outputs are archived in `evaluation_results/`.
