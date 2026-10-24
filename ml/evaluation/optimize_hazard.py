"""
Hazard detection parameter optimization on the VALIDATION set only.

Searches over:
- Canny thresholds
- Morphological kernel sizes and iterations
- Multi-feature hazard detection (Canny + Sobel + Laplacian + local contrast + shadow + top-hat)
- Minimum connected-component area
- LZ scoring weights

Uses F1 and IoU as primary optimization metrics.
The 300-image TEST set is NEVER touched here.
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import torch
from itertools import product

from src.model import LunarUNet
from src.metrics import calculate_hazard_metrics
from scipy import ndimage


def load_model(checkpoint, device):
    model = LunarUNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


def run_inference(model, lr_img, device):
    tensor = torch.from_numpy(lr_img).unsqueeze(0).unsqueeze(0).float() / 255.0
    tensor = tensor.to(device)
    with torch.inference_mode():
        pred = model(tensor)
    return (pred.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)


def build_reference_mask(hr_img):
    """Same reference mask as evaluate_model.py — derived from HR ground truth."""
    k_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_safety = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    thresh_val, _ = cv2.threshold(hr_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    shadow_mask = hr_img < max(25, thresh_val * 0.5)
    top_hat = cv2.morphologyEx(hr_img, cv2.MORPH_TOPHAT, k_tophat)
    black_hat = cv2.morphologyEx(hr_img, cv2.MORPH_BLACKHAT, k_tophat)
    morph_grad = cv2.morphologyEx(hr_img, cv2.MORPH_GRADIENT, k_grad)

    ref_hazard = shadow_mask | (top_hat > 25) | (black_hat > 25) | (morph_grad > 35)
    ref_hazard_mask = cv2.dilate(ref_hazard.astype(np.uint8), k_safety, iterations=1)
    return (ref_hazard_mask > 0).astype(np.uint8)


def predict_hazard_canny_only(pred_img, canny_low, canny_high, kernel_size, iterations):
    """Baseline approach: Canny + dilation."""
    edges = cv2.Canny(pred_img, canny_low, canny_high)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(edges, k, iterations=iterations)
    return (dilated > 0).astype(np.uint8)


def predict_hazard_multi(pred_img, canny_low, canny_high, kernel_size, iterations,
                          sobel_thresh, lap_thresh, shadow_thresh, tophat_thresh,
                          w_canny=0.3, w_sobel=0.2, w_lap=0.15, w_shadow=0.2, w_tophat=0.15,
                          score_thresh=0.35):
    """
    Multi-feature hazard detection combining multiple image features.
    Each feature produces a [0,1] score map, combined with weights, thresholded.
    """
    h, w = pred_img.shape

    # 1. Canny edges (binary → dilated)
    edges = cv2.Canny(pred_img, canny_low, canny_high)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    canny_map = cv2.dilate(edges, k, iterations=iterations).astype(np.float32) / 255.0

    # 2. Sobel gradient magnitude
    sobel_x = cv2.Sobel(pred_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(pred_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_map = (sobel_mag > sobel_thresh).astype(np.float32)

    # 3. Laplacian (second derivative, captures rapid intensity changes)
    lap = np.abs(cv2.Laplacian(pred_img, cv2.CV_64F))
    lap_map = (lap > lap_thresh).astype(np.float32)

    # 4. Shadow regions (very dark areas likely indicate deep shadows/craters)
    shadow_map = (pred_img < shadow_thresh).astype(np.float32)

    # 5. Top-hat (bright features on dark background — crater rims, boulders)
    k_th = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(pred_img, cv2.MORPH_TOPHAT, k_th)
    blackhat = cv2.morphologyEx(pred_img, cv2.MORPH_BLACKHAT, k_th)
    tophat_map = ((tophat > tophat_thresh) | (blackhat > tophat_thresh)).astype(np.float32)

    # Weighted combination
    combined = (w_canny * canny_map + w_sobel * sobel_map + w_lap * lap_map +
                w_shadow * shadow_map + w_tophat * tophat_map)

    # Threshold to binary
    hazard_mask = (combined >= score_thresh).astype(np.uint8)

    # Light morphological cleanup
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hazard_mask = cv2.morphologyEx(hazard_mask, cv2.MORPH_CLOSE, k_close)

    return hazard_mask


def evaluate_config(model, val_files, hr_dir, lr_dir, device, predict_fn, desc=""):
    """Evaluate a hazard config on validation set. Returns mean metrics."""
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0

    for fname in val_files:
        hr_img = cv2.imread(os.path.join(hr_dir, fname), cv2.IMREAD_GRAYSCALE)
        lr_img = cv2.imread(os.path.join(lr_dir, fname), cv2.IMREAD_GRAYSCALE)
        if hr_img is None or lr_img is None:
            continue

        pred_img = run_inference(model, lr_img, device)
        ref_mask = build_reference_mask(hr_img)
        pred_mask = predict_fn(pred_img)

        hm = calculate_hazard_metrics(pred_mask, ref_mask)
        total_tp += hm["tp"]
        total_tn += hm["tn"]
        total_fp += hm["fp"]
        total_fn += hm["fn"]

    total = total_tp + total_tn + total_fp + total_fn
    if total == 0:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "iou": 0}

    acc = (total_tp + total_tn) / total
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4),
        "tp": total_tp,
        "tn": total_tn,
        "fp": total_fp,
        "fn": total_fn,
    }


def optimize(checkpoint="weights/lunar_unet_v2_best.pth"):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint}")

    model = load_model(checkpoint, device)

    # Load validation split
    with open("data/processed_pairs/split.json") as f:
        split = json.load(f)

    val_files = split["val_files"]
    hr_dir = "data/processed_pairs/hr"
    lr_dir = "data/processed_pairs/lr"
    print(f"Validation set: {len(val_files)} images\n")

    results = []

    # === Phase 1: Canny-only parameter sweep ===
    print("=" * 60)
    print("Phase 1: Canny-only parameter sweep")
    print("=" * 60)

    canny_configs = [
        (30, 100), (40, 120), (50, 150), (60, 160), (70, 180),
        (80, 200), (30, 80), (40, 100), (50, 120), (60, 150),
    ]
    kernel_sizes = [5, 7, 9, 11]
    iteration_counts = [1, 2, 3]

    best_canny = {"f1": 0}
    total_configs = len(canny_configs) * len(kernel_sizes) * len(iteration_counts)
    config_idx = 0

    for (cl, ch), ks, it in product(canny_configs, kernel_sizes, iteration_counts):
        config_idx += 1
        fn = lambda img, _cl=cl, _ch=ch, _ks=ks, _it=it: predict_hazard_canny_only(img, _cl, _ch, _ks, _it)
        m = evaluate_config(model, val_files, hr_dir, lr_dir, device, fn)
        entry = {"method": "canny", "canny_low": cl, "canny_high": ch, "kernel": ks, "iter": it, **m}
        results.append(entry)

        if m["f1"] > best_canny["f1"]:
            best_canny = entry
            print(f"  [{config_idx}/{total_configs}] NEW BEST Canny({cl},{ch}) k={ks} it={it}: "
                  f"F1={m['f1']:.4f} IoU={m['iou']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}")

        if config_idx % 20 == 0:
            print(f"  [{config_idx}/{total_configs}] scanning...")

    print(f"\nBest Canny-only: {best_canny}\n")

    # === Phase 2: Multi-feature sweep ===
    print("=" * 60)
    print("Phase 2: Multi-feature hazard detection sweep")
    print("=" * 60)

    # Use best canny params as starting point
    bc_low = best_canny.get("canny_low", 50)
    bc_high = best_canny.get("canny_high", 150)
    bc_kernel = best_canny.get("kernel", 7)
    bc_iter = best_canny.get("iter", 2)

    # Search over additional feature thresholds and weights
    multi_configs = [
        # (sobel_thresh, lap_thresh, shadow_thresh, tophat_thresh, score_thresh)
        (30, 15, 40, 20, 0.25),
        (30, 15, 40, 20, 0.30),
        (30, 15, 40, 20, 0.35),
        (40, 20, 50, 25, 0.25),
        (40, 20, 50, 25, 0.30),
        (40, 20, 50, 25, 0.35),
        (25, 12, 35, 18, 0.25),
        (25, 12, 35, 18, 0.30),
        (50, 25, 45, 30, 0.30),
        (50, 25, 45, 30, 0.35),
        (35, 18, 45, 22, 0.25),
        (35, 18, 45, 22, 0.30),
        (35, 18, 45, 22, 0.35),
        (20, 10, 30, 15, 0.20),
        (20, 10, 30, 15, 0.25),
    ]

    weight_configs = [
        (0.3, 0.2, 0.15, 0.2, 0.15),
        (0.25, 0.25, 0.15, 0.2, 0.15),
        (0.2, 0.2, 0.2, 0.2, 0.2),
        (0.35, 0.15, 0.15, 0.2, 0.15),
        (0.25, 0.2, 0.1, 0.25, 0.2),
    ]

    best_multi = {"f1": 0}
    total_multi = len(multi_configs) * len(weight_configs)
    m_idx = 0

    for (st, lt, sht, tht, sct), (wc, ws, wl, wsh, wth) in product(multi_configs, weight_configs):
        m_idx += 1
        fn = lambda img, _st=st, _lt=lt, _sht=sht, _tht=tht, _sct=sct, _wc=wc, _ws=ws, _wl=wl, _wsh=wsh, _wth=wth: (
            predict_hazard_multi(img, bc_low, bc_high, bc_kernel, bc_iter,
                                 _st, _lt, _sht, _tht, _wc, _ws, _wl, _wsh, _wth, _sct)
        )
        m = evaluate_config(model, val_files, hr_dir, lr_dir, device, fn)
        entry = {
            "method": "multi", "canny_low": bc_low, "canny_high": bc_high,
            "kernel": bc_kernel, "iter": bc_iter,
            "sobel_thresh": st, "lap_thresh": lt, "shadow_thresh": sht,
            "tophat_thresh": tht, "score_thresh": sct,
            "w_canny": wc, "w_sobel": ws, "w_lap": wl, "w_shadow": wsh, "w_tophat": wth,
            **m,
        }
        results.append(entry)

        if m["f1"] > best_multi["f1"]:
            best_multi = entry
            print(f"  [{m_idx}/{total_multi}] NEW BEST Multi: "
                  f"F1={m['f1']:.4f} IoU={m['iou']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}")

        if m_idx % 15 == 0:
            print(f"  [{m_idx}/{total_multi}] scanning...")

    print(f"\nBest multi-feature: {best_multi}\n")

    # === Select overall best ===
    overall_best = best_canny if best_canny["f1"] >= best_multi["f1"] else best_multi
    print("=" * 60)
    print("OVERALL BEST CONFIGURATION")
    print("=" * 60)
    for k, v in overall_best.items():
        print(f"  {k}: {v}")

    # Save results
    output = {
        "checkpoint": checkpoint,
        "val_count": len(val_files),
        "best_canny_only": best_canny,
        "best_multi_feature": best_multi,
        "overall_best": overall_best,
        "all_results": results,
    }

    os.makedirs("evaluation_results", exist_ok=True)
    with open("evaluation_results/hazard_optimization.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to evaluation_results/hazard_optimization.json")
    return overall_best


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "weights/lunar_unet_v2_best.pth"
    optimize(ckpt)
