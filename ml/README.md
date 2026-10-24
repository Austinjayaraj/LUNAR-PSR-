# Machine Learning & Hazard Engine — Lunar PSR Terminal

This directory contains the machine learning core, neural network architectures, training pipelines, hazard analysis experiments, and evaluation benchmarking suites.

## Architecture: LunarUNet
- Custom U-Net with Residual Convolutional Blocks, 4 downsampling stages, and 4 upsampling stages with skip connections.
- Parameter count: **8,559,617**.
- Trained with combined loss: $L = \text{L1} + \text{SSIM} + \text{EdgeLoss}$ (Sobel).
- Active best checkpoint: `weights/lunar_unet_v2_best.pth` (Epoch 57, validation loss: 0.1146).

## Directory Structure
- `src/`: Core Python modules for ML.
  - `model.py`: `LunarUNet` neural network architecture.
  - `dataset.py` & `dataset_ohrc.py`: PyTorch `Dataset` loaders for paired raw/degraded lunar imagery.
  - `losses.py`: Combined L1 + SSIM + Sobel edge loss implementation.
  - `metrics.py`: Rigorous calculation of restoration (MAE, RMSE, PSNR, SSIM, R²) and hazard detection metrics (Precision, Recall, F1, IoU).
  - `preprocess.py`: PDS4 Chandrayaan-2 OHRC ingestion and patch extraction pipeline.
  - `split_dataset.py`: Train/Val/Test deterministic partitioner.
- `training/`:
  - `train.py`: V1 baseline training loop (MSE + VGG perceptual).
  - `train_v2.py`: V2 enhanced training loop (L1 + SSIM + Edge, Cosine LR scheduler, early stopping).
- `evaluation/`:
  - `evaluate_v2_final.py`: Full evaluation on untouched 300-image held-out test set comparing Baseline vs. V1 Improved vs. V2 Multiscale.
  - `evaluate_improved.py`: Pipeline evaluation comparing baseline and improved models.
  - `evaluate_model.py`: Ground truth vs restored evaluation.
  - `optimize_hazard.py`: Validation set optimization for Canny + morphological thresholds.
- `experiments/`:
  - `hazard_v2_optimizer.py`: Multiscale Sobel + morphological edge weight search.
  - `fn_analysis.py`: False-negative hazard distribution analysis.
  - `EXPERIMENT_LOG.json`: Chronological audit log of all hyperparameter sweeps and test runs.

## Usage
Run evaluation from repository root:
```bash
./venv/bin/python ml/evaluation/evaluate_v2_final.py
```
Or use the convenience root forwarding wrappers (`python evaluate_v2_final.py`).
