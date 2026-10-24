"""
Final evaluation on the untouched 300-image test set.

Compares baseline vs improved pipeline:
- Restoration metrics: MAE, RMSE, R², PSNR, SSIM
- Hazard detection metrics: Accuracy, Precision, Recall, F1, IoU
- Efficiency: inference time, parameters, model size
- Visual error analysis: LR, HR, baseline pred, improved pred, hazard masks, FP, FN

IMPORTANT: This script touches the test set EXACTLY ONCE for the final comparison.
"""

import os
import sys
import json
import time
import csv
import cv2
import numpy as np
import torch

from src.model import LunarUNet
from src.metrics import calculate_restoration_metrics, calculate_hazard_metrics


def load_model(checkpoint, device):
    model = LunarUNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


def run_inference_single(model, lr_img, device):
    tensor = torch.from_numpy(lr_img).unsqueeze(0).unsqueeze(0).float() / 255.0
    tensor = tensor.to(device)
    with torch.inference_mode():
        pred = model(tensor)
    return (pred.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)


def build_reference_mask(hr_img):
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


def baseline_hazard(pred_img):
    """Original pipeline: Canny(80,200) + 11x11 ellipse dilation x2."""
    edges = cv2.Canny(pred_img, 80, 200)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilated = cv2.dilate(edges, k, iterations=2)
    return (dilated > 0).astype(np.uint8)


def build_improved_hazard_fn(config):
    """Build hazard function from optimized config."""
    method = config.get("method", "canny")

    if method == "multi":
        def fn(pred_img):
            cl, ch = config["canny_low"], config["canny_high"]
            ks, it = config["kernel"], config["iter"]
            h, w = pred_img.shape

            edges = cv2.Canny(pred_img, cl, ch)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
            canny_map = cv2.dilate(edges, k, iterations=it).astype(np.float32) / 255.0

            sobel_x = cv2.Sobel(pred_img, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(pred_img, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            sobel_map = (sobel_mag > config["sobel_thresh"]).astype(np.float32)

            lap = np.abs(cv2.Laplacian(pred_img, cv2.CV_64F))
            lap_map = (lap > config["lap_thresh"]).astype(np.float32)

            shadow_map = (pred_img < config["shadow_thresh"]).astype(np.float32)

            k_th = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            tophat = cv2.morphologyEx(pred_img, cv2.MORPH_TOPHAT, k_th)
            blackhat = cv2.morphologyEx(pred_img, cv2.MORPH_BLACKHAT, k_th)
            tophat_map = ((tophat > config["tophat_thresh"]) |
                          (blackhat > config["tophat_thresh"])).astype(np.float32)

            combined = (config["w_canny"] * canny_map + config["w_sobel"] * sobel_map +
                        config["w_lap"] * lap_map + config["w_shadow"] * shadow_map +
                        config["w_tophat"] * tophat_map)

            hazard_mask = (combined >= config["score_thresh"]).astype(np.uint8)
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            hazard_mask = cv2.morphologyEx(hazard_mask, cv2.MORPH_CLOSE, k_close)
            return hazard_mask
        return fn
    else:
        def fn(pred_img):
            edges = cv2.Canny(pred_img, config["canny_low"], config["canny_high"])
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config["kernel"], config["kernel"]))
            dilated = cv2.dilate(edges, k, iterations=config["iter"])
            return (dilated > 0).astype(np.uint8)
        return fn


def save_error_analysis(output_dir, sample_data, idx):
    """Generate comprehensive visual error analysis for a single sample."""
    fname = sample_data["filename"]
    lr = sample_data["lr_img"]
    hr = sample_data["hr_img"]
    base_pred = sample_data["baseline_pred"]
    imp_pred = sample_data["improved_pred"]
    ref_mask = sample_data["ref_mask"]
    base_haz = sample_data["baseline_hazard"]
    imp_haz = sample_data["improved_hazard"]

    h, w = hr.shape

    # FP/FN maps for improved
    fp_map = np.zeros((h, w, 3), dtype=np.uint8)
    fn_map = np.zeros((h, w, 3), dtype=np.uint8)
    fp_pixels = (imp_haz > 0) & (ref_mask == 0)
    fn_pixels = (imp_haz == 0) & (ref_mask > 0)
    fp_map[fp_pixels] = [0, 0, 255]  # Red = false positive
    fn_map[fn_pixels] = [255, 0, 0]  # Blue = false negative

    # FP/FN for baseline
    base_fp = np.zeros((h, w, 3), dtype=np.uint8)
    base_fn = np.zeros((h, w, 3), dtype=np.uint8)
    base_fp[(base_haz > 0) & (ref_mask == 0)] = [0, 0, 255]
    base_fn[(base_haz == 0) & (ref_mask > 0)] = [255, 0, 0]

    def to_bgr(gray):
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def mask_to_bgr(mask):
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        vis[mask > 0] = [0, 200, 200]
        return vis

    # Row 1: LR Input | HR Reference | Baseline Pred | Improved Pred
    row1 = np.hstack([to_bgr(lr), to_bgr(hr), to_bgr(base_pred), to_bgr(imp_pred)])

    # Row 2: Ref Hazard | Baseline Hazard | Improved Hazard | Difference (imp - base)
    diff_abs = np.abs(imp_pred.astype(np.float32) - hr.astype(np.float32))
    diff_norm = np.clip(diff_abs * 2, 0, 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    row2 = np.hstack([mask_to_bgr(ref_mask), mask_to_bgr(base_haz), mask_to_bgr(imp_haz), diff_color])

    # Row 3: Baseline FP | Baseline FN | Improved FP | Improved FN
    row3 = np.hstack([base_fp, base_fn, fp_map, fn_map])

    # Add labels
    labels_r1 = ["LR Input", "HR Reference", "Baseline U-Net", "Improved U-Net"]
    labels_r2 = ["Ref Hazard Mask", "Baseline Hazard", "Improved Hazard", "Abs Error (Improved)"]
    labels_r3 = ["Baseline FP (red)", "Baseline FN (blue)", "Improved FP (red)", "Improved FN (blue)"]

    for row, labels in [(row1, labels_r1), (row2, labels_r2), (row3, labels_r3)]:
        for i, label in enumerate(labels):
            x_off = i * w + 5
            cv2.rectangle(row, (x_off, 0), (x_off + len(label) * 8 + 10, 20), (0, 0, 0), -1)
            cv2.putText(row, label, (x_off + 5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Header
    bm = sample_data["baseline_metrics"]
    im = sample_data["improved_metrics"]
    header = np.zeros((30, row1.shape[1], 3), dtype=np.uint8)
    header_text = (f"{fname} | Baseline PSNR={bm['psnr']:.1f} SSIM={bm['ssim']:.3f} | "
                   f"Improved PSNR={im['psnr']:.1f} SSIM={im['ssim']:.3f}")
    cv2.putText(header, header_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

    full = np.vstack([header, row1, row2, row3])
    path = os.path.join(output_dir, f"error_analysis_{idx:03d}_{fname}")
    cv2.imwrite(path, full)
    return path


def measure_efficiency(model, device, n_warmup=3, n_measure=10):
    """Measure inference time for a single 256x256 image."""
    dummy = torch.randn(1, 1, 256, 256).to(device)

    with torch.inference_mode():
        for _ in range(n_warmup):
            model(dummy)

        if device.type == "mps":
            torch.mps.synchronize()

        times = []
        for _ in range(n_measure):
            t0 = time.perf_counter()
            model(dummy)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    return {
        "mean_ms": round(float(np.mean(times)), 2),
        "std_ms": round(float(np.std(times)), 2),
        "min_ms": round(float(np.min(times)), 2),
        "max_ms": round(float(np.max(times)), 2),
    }


def evaluate_final(baseline_ckpt="weights/lunar_unet_best.pth",
                    improved_ckpt="weights/lunar_unet_v2_best.pth",
                    hazard_config_path="evaluation_results/hazard_optimization.json",
                    output_dir="evaluation_results/final"):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visual_analysis")
    os.makedirs(vis_dir, exist_ok=True)

    # Load models
    print("Loading baseline model...")
    baseline_model = load_model(baseline_ckpt, device)
    print("Loading improved model...")
    improved_model = load_model(improved_ckpt, device)

    # Load optimized hazard config
    improved_hazard_fn = baseline_hazard  # fallback
    hazard_config = None
    if os.path.exists(hazard_config_path):
        with open(hazard_config_path) as f:
            opt = json.load(f)
        hazard_config = opt.get("overall_best", None)
        if hazard_config:
            improved_hazard_fn = build_improved_hazard_fn(hazard_config)
            print(f"Using optimized hazard config: method={hazard_config.get('method')}")
    else:
        print("No hazard optimization found, using baseline hazard params for improved model too.")

    # Load test split
    with open("data/processed_pairs/split.json") as f:
        split = json.load(f)
    test_files = split["test_files"]
    hr_dir = "data/processed_pairs/hr"
    lr_dir = "data/processed_pairs/lr"
    print(f"Test set: {len(test_files)} images (UNTOUCHED until now)\n")

    # Run evaluation
    baseline_rest = []
    improved_rest = []
    baseline_haz_agg = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    improved_haz_agg = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    all_samples = []

    for idx, fname in enumerate(test_files):
        hr_img = cv2.imread(os.path.join(hr_dir, fname), cv2.IMREAD_GRAYSCALE)
        lr_img = cv2.imread(os.path.join(lr_dir, fname), cv2.IMREAD_GRAYSCALE)
        if hr_img is None or lr_img is None:
            continue

        # Baseline
        base_pred = run_inference_single(baseline_model, lr_img, device)
        base_rest_m = calculate_restoration_metrics(base_pred, hr_img)
        base_haz_mask = baseline_hazard(base_pred)
        ref_mask = build_reference_mask(hr_img)
        base_haz_m = calculate_hazard_metrics(base_haz_mask, ref_mask)

        # Improved
        imp_pred = run_inference_single(improved_model, lr_img, device)
        imp_rest_m = calculate_restoration_metrics(imp_pred, hr_img)
        imp_haz_mask = improved_hazard_fn(imp_pred)
        imp_haz_m = calculate_hazard_metrics(imp_haz_mask, ref_mask)

        baseline_rest.append(base_rest_m)
        improved_rest.append(imp_rest_m)

        for k in baseline_haz_agg:
            baseline_haz_agg[k] += base_haz_m[k]
            improved_haz_agg[k] += imp_haz_m[k]

        all_samples.append({
            "filename": fname,
            "lr_img": lr_img,
            "hr_img": hr_img,
            "baseline_pred": base_pred,
            "improved_pred": imp_pred,
            "ref_mask": ref_mask,
            "baseline_hazard": base_haz_mask,
            "improved_hazard": imp_haz_mask,
            "baseline_metrics": base_rest_m,
            "improved_metrics": imp_rest_m,
            "baseline_haz_metrics": base_haz_m,
            "improved_haz_metrics": imp_haz_m,
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_files):
            print(f"  Processed {idx + 1}/{len(test_files)}...")

    # Generate visual error analysis for representative samples
    # Sort by improvement in F1
    for s in all_samples:
        s["f1_improvement"] = s["improved_haz_metrics"]["f1"] - s["baseline_haz_metrics"]["f1"]

    sorted_samples = sorted(all_samples, key=lambda x: x["f1_improvement"], reverse=True)

    # Select: best improvement, worst improvement, median, 2 random
    vis_indices = [0, len(sorted_samples) // 4, len(sorted_samples) // 2,
                   3 * len(sorted_samples) // 4, len(sorted_samples) - 1]
    # Also add a few from the middle of the original ordering
    extra = [10, 50, 150]
    vis_indices = sorted(set(vis_indices + extra))

    print(f"\nGenerating visual error analysis for {len(vis_indices)} samples...")
    for i, vi in enumerate(vis_indices):
        if vi < len(sorted_samples):
            save_error_analysis(vis_dir, sorted_samples[vi], i)

    # Compute aggregate metrics
    def agg_restoration(results_list):
        keys = ["mae", "rmse", "r2", "psnr", "ssim"]
        out = {}
        for k in keys:
            vals = [r[k] for r in results_list if r[k] != float("inf")]
            out[k] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}
        return out

    def agg_hazard(agg):
        tp, tn, fp, fn = agg["tp"], agg["tn"], agg["fp"], agg["fn"]
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        return {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1": round(f1 * 100, 2),
            "iou": round(iou * 100, 2),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        }

    base_rest_agg = agg_restoration(baseline_rest)
    imp_rest_agg = agg_restoration(improved_rest)
    base_haz_final = agg_hazard(baseline_haz_agg)
    imp_haz_final = agg_hazard(improved_haz_agg)

    # Efficiency
    print("\nMeasuring inference efficiency...")
    mps_base = measure_efficiency(baseline_model, device)
    mps_imp = measure_efficiency(improved_model, device)

    cpu_device = torch.device("cpu")
    baseline_cpu = load_model(baseline_ckpt, cpu_device)
    improved_cpu = load_model(improved_ckpt, cpu_device)
    cpu_base = measure_efficiency(baseline_cpu, cpu_device)
    cpu_imp = measure_efficiency(improved_cpu, cpu_device)

    params = sum(p.numel() for p in improved_model.parameters())
    ckpt_size_mb = os.path.getsize(improved_ckpt) / (1024 * 1024)
    base_ckpt_size_mb = os.path.getsize(baseline_ckpt) / (1024 * 1024)

    # Print comparison table
    print("\n" + "=" * 75)
    print("FINAL EVALUATION RESULTS — 300-image test set")
    print("=" * 75)

    print("\n--- HAZARD DETECTION ---")
    print(f"{'Metric':<12} {'Baseline':>12} {'Improved':>12} {'Change':>12}")
    print("-" * 50)
    for k in ["accuracy", "precision", "recall", "f1", "iou"]:
        bv = base_haz_final[k]
        iv = imp_haz_final[k]
        ch = iv - bv
        print(f"{k.upper():<12} {bv:>11.2f}% {iv:>11.2f}% {ch:>+11.2f}")

    print("\n--- RESTORATION ---")
    print(f"{'Metric':<12} {'Baseline':>12} {'Improved':>12} {'Change':>12}")
    print("-" * 50)
    for k in ["mae", "rmse", "r2", "psnr", "ssim"]:
        bv = base_rest_agg[k]["mean"]
        iv = imp_rest_agg[k]["mean"]
        ch = iv - bv
        unit = " dB" if k == "psnr" else ""
        print(f"{k.upper():<12} {bv:>12.4f}{unit} {iv:>12.4f}{unit} {ch:>+11.4f}")

    print("\n--- EFFICIENCY ---")
    print(f"Parameters:          {params:,}")
    print(f"Model size (base):   {base_ckpt_size_mb:.1f} MB")
    print(f"Model size (improv): {ckpt_size_mb:.1f} MB")
    print(f"MPS inference (base):   {mps_base['mean_ms']:.1f} ms")
    print(f"MPS inference (improv): {mps_imp['mean_ms']:.1f} ms")
    print(f"CPU inference (base):   {cpu_base['mean_ms']:.1f} ms")
    print(f"CPU inference (improv): {cpu_imp['mean_ms']:.1f} ms")

    # Save full results
    full_results = {
        "test_set_size": len(test_files),
        "baseline_checkpoint": baseline_ckpt,
        "improved_checkpoint": improved_ckpt,
        "hazard_config": hazard_config,
        "hazard_detection": {
            "baseline": base_haz_final,
            "improved": imp_haz_final,
        },
        "restoration": {
            "baseline": base_rest_agg,
            "improved": imp_rest_agg,
        },
        "efficiency": {
            "parameters": params,
            "baseline_model_size_mb": round(base_ckpt_size_mb, 1),
            "improved_model_size_mb": round(ckpt_size_mb, 1),
            "mps_inference_baseline": mps_base,
            "mps_inference_improved": mps_imp,
            "cpu_inference_baseline": cpu_base,
            "cpu_inference_improved": cpu_imp,
        },
        "reference_type": "SYNTHETIC (Derived Photometric Reference from 0.25m HR telemetry)",
    }

    with open(os.path.join(output_dir, "final_comparison.json"), "w") as f:
        json.dump(full_results, f, indent=2)

    # Save per-image CSV
    csv_path = os.path.join(output_dir, "per_image_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "base_mae", "base_rmse", "base_r2", "base_psnr", "base_ssim",
            "imp_mae", "imp_rmse", "imp_r2", "imp_psnr", "imp_ssim",
            "base_haz_f1", "base_haz_iou", "base_haz_prec", "base_haz_rec",
            "imp_haz_f1", "imp_haz_iou", "imp_haz_prec", "imp_haz_rec",
        ])
        for s in all_samples:
            bm = s["baseline_metrics"]
            im = s["improved_metrics"]
            bh = s["baseline_haz_metrics"]
            ih = s["improved_haz_metrics"]
            writer.writerow([
                s["filename"],
                bm["mae"], bm["rmse"], bm["r2"], bm["psnr"], bm["ssim"],
                im["mae"], im["rmse"], im["r2"], im["psnr"], im["ssim"],
                bh["f1"], bh["iou"], bh["precision"], bh["recall"],
                ih["f1"], ih["iou"], ih["precision"], ih["recall"],
            ])

    print(f"\nResults saved to {output_dir}/")
    print(f"  final_comparison.json")
    print(f"  per_image_metrics.csv")
    print(f"  visual_analysis/ ({len(vis_indices)} error analysis panels)")
    print("=" * 75)

    return full_results


if __name__ == "__main__":
    evaluate_final()
