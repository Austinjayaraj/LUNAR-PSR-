"""
Evaluation metrics for Lunar PSR Restoration and Hazard Detection.
Strictly calculates scientific metrics without hardcoding or fabricating values.
"""

from typing import Dict, Any
import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr


def calculate_restoration_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    data_range: float = 255.0,
) -> Dict[str, float]:
    """
    Calculate image restoration metrics: MAE, RMSE, R², PSNR, SSIM.

    Args:
        pred: Predicted restoration image (H, W), dtype float or uint8.
        gt: Ground-truth reference image (H, W), same shape as pred.
        data_range: Dynamic range of pixel values (default 255.0 for [0, 255]).

    Returns:
        Dict with keys: mae, rmse, r2, psnr, ssim.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    pred_f = pred.astype(np.float64)
    gt_f = gt.astype(np.float64)

    # 1. MAE
    mae = float(np.mean(np.abs(pred_f - gt_f)))

    # 2. RMSE
    mse = float(np.mean((pred_f - gt_f) ** 2))
    rmse = float(np.sqrt(mse))

    # 3. R² (Coefficient of determination)
    ss_res = float(np.sum((gt_f - pred_f) ** 2))
    gt_mean = float(np.mean(gt_f))
    ss_tot = float(np.sum((gt_f - gt_mean) ** 2))

    if ss_tot < 1e-12:
        # Ground truth has zero variance (flat image)
        r2 = 1.0 if ss_res < 1e-12 else 0.0
    else:
        r2 = float(1.0 - (ss_res / ss_tot))

    # 4. PSNR
    if mse < 1e-12:
        psnr = float("inf")
    else:
        psnr = float(skimage_psnr(gt_f, pred_f, data_range=data_range))

    # 5. SSIM
    if mse < 1e-12:
        ssim = 1.0
    else:
        ssim = float(skimage_ssim(gt_f, pred_f, data_range=data_range))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "psnr": round(psnr, 4) if psnr != float("inf") else float("inf"),
        "ssim": round(ssim, 4),
    }


def calculate_hazard_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculate binary classification metrics for hazard masks:
    Accuracy, Precision, Recall, F1-score, IoU.

    Args:
        pred_mask: Binary prediction mask (H, W), boolean or integer {0, 1}.
        gt_mask: Binary ground-truth mask (H, W), boolean or integer {0, 1}.

    Returns:
        Dict with keys: tp, tn, fp, fn, accuracy, precision, recall, f1, iou.
    """
    # Guard against continuous inputs: require boolean or binary values
    unique_pred = np.unique(pred_mask)
    if not np.all(np.isin(unique_pred, [0, 1, False, True])):
        raise ValueError(
            "Hazard metrics require binary mask (values in {0, 1}), "
            "do not pass continuous outputs directly."
        )

    unique_gt = np.unique(gt_mask)
    if not np.all(np.isin(unique_gt, [0, 1, False, True])):
        raise ValueError(
            "Hazard ground truth must be a binary mask (values in {0, 1})."
        )

    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}")

    pred_bin = (pred_mask > 0).astype(bool)
    gt_bin = (gt_mask > 0).astype(bool)

    tp = int(np.sum(pred_bin & gt_bin))
    tn = int(np.sum((~pred_bin) & (~gt_bin)))
    fp = int(np.sum(pred_bin & (~gt_bin)))
    fn = int(np.sum((~pred_bin) & gt_bin))

    total = tp + tn + fp + fn
    accuracy = float((tp + tn) / total) if total > 0 else 0.0

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    if (precision + recall) > 0:
        f1 = float(2 * precision * recall / (precision + recall))
    else:
        f1 = 0.0

    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4),
    }
