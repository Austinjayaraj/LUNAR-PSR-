"""
Improved U-Net training on real OHRC data with:
- L1 + SSIM + Edge loss (replaces MSE + VGG perceptual)
- Train/val split with validation monitoring
- Early stopping on validation loss
- Cosine annealing LR schedule
- Data augmentation (flips, rotations)
- Reproducible seed
- MPS support
- Per-epoch metric tracking (PSNR, SSIM, MAE, RMSE)
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model import LunarUNet
from src.dataset_ohrc import OHRCDataset
from src.losses import CombinedRestorationLoss
from src.metrics import calculate_restoration_metrics


SEED = 42
BATCH_SIZE = 8
LR = 3e-4
EPOCHS = 80
PATIENCE = 15
DATA_DIR = "data/processed_pairs"
WEIGHTS_DIR = "weights"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate(model, val_loader, criterion, device, compute_metrics=False):
    model.eval()
    total_loss = 0.0
    count = 0
    metrics_accum = {"mae": [], "rmse": [], "psnr": [], "ssim": []}

    with torch.inference_mode():
        for lr_imgs, hr_imgs in val_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            outputs = model(lr_imgs)
            loss, _ = criterion(outputs, hr_imgs)
            total_loss += loss.item() * lr_imgs.size(0)
            count += lr_imgs.size(0)

            if compute_metrics:
                preds = (outputs.squeeze(1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                targets = (hr_imgs.squeeze(1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                for p, t in zip(preds, targets):
                    m = calculate_restoration_metrics(p, t)
                    for k in metrics_accum:
                        metrics_accum[k].append(m[k])

    avg_loss = total_loss / count if count > 0 else float("inf")
    if compute_metrics and metrics_accum["mae"]:
        avg_metrics = {k: float(np.mean(v)) for k, v in metrics_accum.items()}
    else:
        avg_metrics = {"mae": 0, "rmse": 0, "psnr": 0, "ssim": 0}
    return avg_loss, avg_metrics


def train():
    set_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Seed: {SEED}")

    # Datasets
    train_ds = OHRCDataset(DATA_DIR, split="train", augment=True)
    val_ds = OHRCDataset(DATA_DIR, split="val", augment=False)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = LunarUNet(in_channels=1, out_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Loss, optimizer, scheduler
    criterion = CombinedRestorationLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    print(f"\nTraining for up to {EPOCHS} epochs (patience={PATIENCE})\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_count = 0
        component_totals = {"pixel": 0.0, "ssim": 0.0, "edge": 0.0}

        for lr_imgs, hr_imgs in train_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            optimizer.zero_grad()
            outputs = model(lr_imgs)
            loss, components = criterion(outputs, hr_imgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = lr_imgs.size(0)
            train_loss += loss.item() * bs
            train_count += bs
            for k in component_totals:
                component_totals[k] += components[k] * bs

        scheduler.step()
        avg_train = train_loss / train_count
        avg_components = {k: v / train_count for k, v in component_totals.items()}

        # Validation (full metrics every 10 epochs or last epoch for speed)
        do_metrics = (epoch % 10 == 0) or (epoch == EPOCHS) or (patience_counter >= PATIENCE - 1)
        val_loss, val_metrics = validate(model, val_loader, criterion, device, compute_metrics=do_metrics)
        elapsed = time.time() - t0

        # Checkpointing
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "lunar_unet_v2_best.pth"))
        else:
            patience_counter += 1

        entry = {
            "epoch": epoch,
            "train_loss": round(avg_train, 6),
            "val_loss": round(val_loss, 6),
            "val_psnr": round(val_metrics["psnr"], 2),
            "val_ssim": round(val_metrics["ssim"], 4),
            "val_mae": round(val_metrics["mae"], 2),
            "val_rmse": round(val_metrics["rmse"], 2),
            "lr": scheduler.get_last_lr()[0],
            "time_s": round(elapsed, 1),
            "pixel_loss": round(avg_components["pixel"], 6),
            "ssim_loss": round(avg_components["ssim"], 6),
            "edge_loss": round(avg_components["edge"], 6),
        }
        history.append(entry)

        marker = "*" if improved else ""
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train: {avg_train:.5f} | Val: {val_loss:.5f} | "
            f"PSNR: {val_metrics['psnr']:.2f} | SSIM: {val_metrics['ssim']:.4f} | "
            f"MAE: {val_metrics['mae']:.1f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.1f}s {marker}"
        )

        sys.stdout.flush()

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    # Save training history
    with open(os.path.join(WEIGHTS_DIR, "training_history_v2.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Also save final checkpoint
    torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "lunar_unet_v2_final.pth"))

    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Best checkpoint: {WEIGHTS_DIR}/lunar_unet_v2_best.pth")
    print(f"Training history: {WEIGHTS_DIR}/training_history_v2.json")


if __name__ == "__main__":
    train()
