"""
Evaluate Lunar PSR U-Net Restoration Model and Hazard Detection Pipeline.
Calculates actual performance metrics on held-out test data without fabrication.
"""

import os
import sys
import json
import csv
import argparse
from datetime import datetime
import cv2
import numpy as np
import torch

from src.model import LunarUNet
from src.metrics import calculate_restoration_metrics, calculate_hazard_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Lunar PSR Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weights/lunar_unet_best.pth",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed_pairs",
        help="Path to processed pairs directory containing hr/ and lr/",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of dataset held out for testing (default: 0.20 = 20%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/test split",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--num-visualizations",
        type=int,
        default=5,
        help="Number of test samples to generate side-by-side visualizations for",
    )
    return parser.parse_args()


def load_dataset_pairs(data_dir: str, test_ratio: float = 0.2, seed: int = 42):
    """
    Load paired LR and HR images and split into train and held-out test sets.
    """
    hr_dir = os.path.join(data_dir, "hr")
    lr_dir = os.path.join(data_dir, "lr")

    if not os.path.exists(hr_dir) or not os.path.exists(lr_dir):
        raise FileNotFoundError(f"Missing hr or lr directory inside {data_dir}")

    # Find common image files
    hr_files = {f for f in os.listdir(hr_dir) if f.endswith(".png")}
    lr_files = {f for f in os.listdir(lr_dir) if f.endswith(".png")}
    common_files = sorted(list(hr_files.intersection(lr_files)), key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)

    total_count = len(common_files)
    if total_count == 0:
        raise ValueError(f"No paired images found in {data_dir}")

    # Deterministic split
    rng = np.random.RandomState(seed)
    indices = np.arange(total_count)
    rng.shuffle(indices)

    test_count = max(1, int(total_count * test_ratio))
    test_indices = sorted(indices[:test_count].tolist())
    train_indices = sorted(indices[test_count:].tolist())

    test_files = [common_files[i] for i in test_indices]

    return {
        "all_files": common_files,
        "test_files": test_files,
        "test_count": len(test_files),
        "train_count": len(train_indices),
        "total_count": total_count,
        "hr_dir": hr_dir,
        "lr_dir": lr_dir,
    }


def save_visualization(
    output_path: str,
    lr_img: np.ndarray,
    gt_img: np.ndarray,
    pred_img: np.ndarray,
    metrics: dict,
    sample_name: str,
):
    """
    Create a 4-panel comparison: Input (LR) | Ground Truth (HR) | Prediction | Absolute Error.
    """
    # Absolute error map
    diff = np.abs(pred_img.astype(np.float32) - gt_img.astype(np.float32))
    diff_norm = np.clip(diff * 2.0, 0, 255).astype(np.uint8)  # enhanced for visibility
    diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    # Convert grayscale images to BGR for concatenation
    lr_bgr = cv2.cvtColor(lr_img, cv2.COLOR_GRAY2BGR)
    gt_bgr = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2BGR)
    pred_bgr = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)

    # Add text labels on top of each tile
    panels = [
        (lr_bgr, "Input (LR)"),
        (gt_bgr, "Ground Truth (HR)"),
        (pred_bgr, f"Pred (MAE={metrics['mae']:.1f})"),
        (diff_color, f"Abs Error (RMSE={metrics['rmse']:.1f})"),
    ]

    labeled_panels = []
    for panel, label in panels:
        p_copy = panel.copy()
        cv2.rectangle(p_copy, (0, 0), (256, 26), (0, 0, 0), -1)
        cv2.putText(
            p_copy,
            label,
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        labeled_panels.append(p_copy)

    # Concatenate horizontally
    combined = np.hstack(labeled_panels)

    # Add header banner with sample details
    header = np.zeros((36, combined.shape[1], 3), dtype=np.uint8)
    banner_text = (
        f"Sample: {sample_name} | PSNR: {metrics['psnr']:.2f} dB | "
        f"SSIM: {metrics['ssim']:.4f} | R2: {metrics['r2']:.4f}"
    )
    cv2.putText(
        header,
        banner_text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )

    final_image = np.vstack([header, combined])
    cv2.imwrite(output_path, final_image)


def evaluate(args):
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = "CUDA"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "MPS"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    print("=" * 65)
    print("🌙 LUNAR PSR MODEL EVALUATION SYSTEM")
    print("=" * 65)
    print(f"Device:           {device_name} ({device})")
    print(f"Checkpoint:       {args.checkpoint}")
    print(f"Dataset path:     {args.data_dir}")

    # Check checkpoint existence
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    # Load dataset split
    dataset_info = load_dataset_pairs(args.data_dir, test_ratio=args.test_ratio, seed=args.seed)
    test_files = dataset_info["test_files"]
    print(f"Total dataset:    {dataset_info['total_count']} pairs")
    print(f"Held-out test:    {dataset_info['test_count']} samples (ratio: {args.test_ratio:.0%})")
    print(f"Train/Val split:  {dataset_info['train_count']} samples")

    # Load Model
    model = LunarUNet(in_channels=1, out_channels=1).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model successfully loaded into memory.\n")

    # Setup output directories
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Run inference and calculate per-image metrics
    per_image_results = []
    hazard_metrics_list = []
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    k_pred = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_safety = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    print(f"Running inference and hazard evaluation on {len(test_files)} test samples...")

    for idx, fname in enumerate(test_files):
        hr_path = os.path.join(dataset_info["hr_dir"], fname)
        lr_path = os.path.join(dataset_info["lr_dir"], fname)

        hr_img = cv2.imread(hr_path, cv2.IMREAD_GRAYSCALE)
        lr_img = cv2.imread(lr_path, cv2.IMREAD_GRAYSCALE)

        if hr_img is None or lr_img is None:
            print(f"⚠️ Warning: skipping unreadable sample {fname}")
            continue

        # 1. Inference: exactly matching existing pipeline in api/main.py
        tensor = torch.from_numpy(lr_img).unsqueeze(0).unsqueeze(0).float() / 255.0
        tensor = tensor.to(device)

        with torch.inference_mode():
            pred_tensor = model(tensor)
            pred_f = pred_tensor.squeeze().cpu().numpy() * 255.0
            pred_img = np.clip(pred_f, 0, 255).astype(np.uint8)

        # 2. Restoration metrics
        rest_m = calculate_restoration_metrics(pred_img, hr_img, data_range=255.0)
        rest_m["filename"] = fname

        # 3. Existing Hazard Prediction (api/main.py: Canny + morphological dilation)
        edges_pred = cv2.Canny(pred_img, 80, 200)
        danger_pred = cv2.dilate(edges_pred, k_pred, iterations=2)
        pred_hazard_mask = (danger_pred > 0).astype(np.uint8)

        # 4. Scientifically Defensible Derived Reference Hazard Mask (from native uncorrupted 0.25m HR telemetry)
        # Uses independent multi-scale morphology + photometric shadow extraction (not Canny)
        thresh_val, _ = cv2.threshold(hr_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        shadow_mask = hr_img < max(25, thresh_val * 0.5)
        top_hat = cv2.morphologyEx(hr_img, cv2.MORPH_TOPHAT, k_tophat)
        black_hat = cv2.morphologyEx(hr_img, cv2.MORPH_BLACKHAT, k_tophat)
        morph_grad = cv2.morphologyEx(hr_img, cv2.MORPH_GRADIENT, k_grad)

        ref_hazard = (shadow_mask) | (top_hat > 25) | (black_hat > 25) | (morph_grad > 35)
        ref_hazard_mask = cv2.dilate(ref_hazard.astype(np.uint8), k_safety, iterations=1)
        ref_hazard_mask = (ref_hazard_mask > 0).astype(np.uint8)

        # 5. Hazard Classification Metrics
        haz_m = calculate_hazard_metrics(pred_hazard_mask, ref_hazard_mask)
        haz_m["filename"] = fname
        hazard_metrics_list.append(haz_m)

        total_tp += haz_m["tp"]
        total_tn += haz_m["tn"]
        total_fp += haz_m["fp"]
        total_fn += haz_m["fn"]

        per_image_results.append({
            "filename": fname,
            "lr_img": lr_img,
            "hr_img": hr_img,
            "pred_img": pred_img,
            "pred_hazard_mask": pred_hazard_mask,
            "ref_hazard_mask": ref_hazard_mask,
            "metrics": rest_m,
            "hazard_metrics": haz_m,
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_files):
            print(f"   Processed {idx + 1}/{len(test_files)} images...")

    if not per_image_results:
        raise RuntimeError("No test images were evaluated.")

    # Aggregate restoration metrics
    maes = [r["metrics"]["mae"] for r in per_image_results]
    rmses = [r["metrics"]["rmse"] for r in per_image_results]
    r2s = [r["metrics"]["r2"] for r in per_image_results]
    psnrs = [r["metrics"]["psnr"] for r in per_image_results if r["metrics"]["psnr"] != float("inf")]
    ssims = [r["metrics"]["ssim"] for r in per_image_results]

    restoration_summary = {
        "MAE": {
            "mean": round(float(np.mean(maes)), 4),
            "std": round(float(np.std(maes)), 4),
            "median": round(float(np.median(maes)), 4),
            "min": round(float(np.min(maes)), 4),
            "max": round(float(np.max(maes)), 4),
        },
        "RMSE": {
            "mean": round(float(np.mean(rmses)), 4),
            "std": round(float(np.std(rmses)), 4),
            "median": round(float(np.median(rmses)), 4),
            "min": round(float(np.min(rmses)), 4),
            "max": round(float(np.max(rmses)), 4),
        },
        "R2": {
            "mean": round(float(np.mean(r2s)), 4),
            "std": round(float(np.std(r2s)), 4),
            "median": round(float(np.median(r2s)), 4),
            "min": round(float(np.min(r2s)), 4),
            "max": round(float(np.max(r2s)), 4),
        },
        "PSNR": {
            "mean": round(float(np.mean(psnrs)), 4) if psnrs else 0.0,
            "std": round(float(np.std(psnrs)), 4) if psnrs else 0.0,
            "median": round(float(np.median(psnrs)), 4) if psnrs else 0.0,
            "min": round(float(np.min(psnrs)), 4) if psnrs else 0.0,
            "max": round(float(np.max(psnrs)), 4) if psnrs else 0.0,
        },
        "SSIM": {
            "mean": round(float(np.mean(ssims)), 4),
            "std": round(float(np.std(ssims)), 4),
            "median": round(float(np.median(ssims)), 4),
            "min": round(float(np.min(ssims)), 4),
            "max": round(float(np.max(ssims)), 4),
        },
    }

    # Aggregate hazard metrics across all 300 test samples
    accs = [m["accuracy"] for m in hazard_metrics_list]
    precs = [m["precision"] for m in hazard_metrics_list]
    recs = [m["recall"] for m in hazard_metrics_list]
    f1s = [m["f1"] for m in hazard_metrics_list]
    ious = [m["iou"] for m in hazard_metrics_list]

    mean_acc = round(float(np.mean(accs)), 4)
    mean_prec = round(float(np.mean(precs)), 4)
    mean_rec = round(float(np.mean(recs)), 4)
    mean_f1 = round(float(np.mean(f1s)), 4)
    mean_iou = round(float(np.mean(ious)), 4)

    hazard_summary = {
        "reference_type": "SYNTHETIC",
        "reference_description": (
            "Derived Photometric Reference Mask extracted from native 0.25m OHRC ground-truth telemetry "
            "using multi-scale Top-Hat/Black-Hat morphology, shadow thresholding, and morphological gradient "
            "(strictly independent from prediction pipeline's Canny detector). Not human ground truth."
        ),
        "mean_metrics": {
            "accuracy": mean_acc,
            "precision": mean_prec,
            "recall": mean_rec,
            "f1": mean_f1,
            "iou": mean_iou,
        },
        "aggregate_confusion_matrix": {
            "tp": total_tp,
            "tn": total_tn,
            "fp": total_fp,
            "fn": total_fn,
            "total_pixels": total_tp + total_tn + total_fp + total_fn,
        },
    }

    # Generate visualizations for select samples: best, median, worst by PSNR
    sorted_by_psnr = sorted(per_image_results, key=lambda x: x["metrics"]["psnr"])
    selected_samples = []
    if len(sorted_by_psnr) > 0:
        selected_samples.append((sorted_by_psnr[0], "worst_sample"))
        mid = len(sorted_by_psnr) // 2
        selected_samples.append((sorted_by_psnr[mid], "median_sample"))
        selected_samples.append((sorted_by_psnr[-1], "best_sample"))
        for i in range(min(2, len(sorted_by_psnr))):
            s = per_image_results[i]
            if s not in [x[0] for x in selected_samples]:
                selected_samples.append((s, f"test_sample_{s['filename']}"))

    for item, tag in selected_samples:
        vis_filename = f"{tag}_{item['filename']}"
        vis_path = os.path.join(vis_dir, vis_filename)
        save_visualization(
            vis_path,
            item["lr_img"],
            item["hr_img"],
            item["pred_img"],
            item["metrics"],
            item["filename"],
        )

    # Save metrics.csv
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "mae", "rmse", "r2", "psnr_db", "ssim", "haz_acc", "haz_prec", "haz_rec", "haz_f1", "haz_iou", "tp", "tn", "fp", "fn"])
        for r in per_image_results:
            m = r["metrics"]
            hm = r["hazard_metrics"]
            writer.writerow([
                m["filename"], m["mae"], m["rmse"], m["r2"], m["psnr"], m["ssim"],
                hm["accuracy"], hm["precision"], hm["recall"], hm["f1"], hm["iou"],
                hm["tp"], hm["tn"], hm["fp"], hm["fn"]
            ])
        writer.writerow([])
        writer.writerow(["OVERALL_RESTORATION_SUMMARY", "MAE", "RMSE", "R2", "PSNR_dB", "SSIM"])
        writer.writerow(["mean", restoration_summary["MAE"]["mean"], restoration_summary["RMSE"]["mean"], restoration_summary["R2"]["mean"], restoration_summary["PSNR"]["mean"], restoration_summary["SSIM"]["mean"]])
        writer.writerow(["std", restoration_summary["MAE"]["std"], restoration_summary["RMSE"]["std"], restoration_summary["R2"]["std"], restoration_summary["PSNR"]["std"], restoration_summary["SSIM"]["std"]])
        writer.writerow(["median", restoration_summary["MAE"]["median"], restoration_summary["RMSE"]["median"], restoration_summary["R2"]["median"], restoration_summary["PSNR"]["median"], restoration_summary["SSIM"]["median"]])
        writer.writerow([])
        writer.writerow(["OVERALL_HAZARD_SUMMARY", "Accuracy", "Precision", "Recall", "F1", "IoU", "Aggregate_TP", "Aggregate_TN", "Aggregate_FP", "Aggregate_FN"])
        writer.writerow(["mean", mean_acc, mean_prec, mean_rec, mean_f1, mean_iou, total_tp, total_tn, total_fp, total_fn])

    # Save metrics.json
    json_path = os.path.join(args.output_dir, "metrics.json")
    results_payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "checkpoint": args.checkpoint,
        "dataset_path": args.data_dir,
        "device": device_name,
        "total_dataset_samples": dataset_info["total_count"],
        "test_samples_count": len(test_files),
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "restoration_metrics": restoration_summary,
        "hazard_detection_metrics": hazard_summary,
        "per_image_metrics_count": len(per_image_results),
        "visualizations_saved": [os.path.basename(f) for f in os.listdir(vis_dir)],
    }

    with open(json_path, "w") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n✅ Results successfully saved to: {args.output_dir}/")
    print(f"   ├── metrics.csv")
    print(f"   ├── metrics.json")
    print(f"   └── visualizations/ ({len(selected_samples)} figures generated)")

    # Print final summary report to stdout
    print("\n" + "=" * 65)
    print("HAZARD DETECTION EVALUATION")
    print("=" * 65)
    print(f"Test Samples: {len(test_files)}")
    print(f"Device:       {device_name}")
    print(f"Accuracy  = {mean_acc:.4f}")
    print(f"Precision = {mean_prec:.4f}")
    print(f"Recall    = {mean_rec:.4f}")
    print(f"F1        = {mean_f1:.4f}")
    print(f"IoU       = {mean_iou:.4f}")
    print(f"TP = {total_tp:,}")
    print(f"TN = {total_tn:,}")
    print(f"FP = {total_fp:,}")
    print(f"FN = {total_fn:,}")
    print("Reference Type:")
    print("SYNTHETIC (Derived Photometric Reference from 0.25m HR telemetry)")
    print("=" * 65)

    return results_payload


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
