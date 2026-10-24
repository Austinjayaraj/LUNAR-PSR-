# Model Checkpoints & Weights — Lunar PSR Terminal

This directory houses the trained PyTorch weights and training histories for the `LunarUNet` restoration model.

## Checkpoint Files
- `lunar_unet_v2_best.pth` (34.3 MB):
  - **Epoch**: 57
  - **Validation Loss**: 0.1146
  - **Loss Objective**: L1 + SSIM + Edge Loss (Sobel)
  - **Architecture**: 8,559,617 parameters
  - **Active Production Checkpoint**: Default checkpoint loaded by the FastAPI backend and inference pipeline.
- `lunar_unet_v2_final.pth` (34.3 MB):
  - Final checkpoint at the conclusion of V2 training schedule.
- `lunar_unet_best.pth` (34.3 MB):
  - Baseline V1 model trained using MSE + VGG perceptual loss.
- `training_history_v2.json`:
  - Full epoch-by-epoch record of training loss, validation loss, PSNR, SSIM, learning rate, and early stopping triggers across all epochs.

## Preservation Note
These are core trained artifacts and must not be deleted or modified during code cleanup.
