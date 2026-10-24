"""
False-negative analysis on validation set.

For representative samples generates visualizations showing:
  A. restored image
  B. reference hazard mask
  C. predicted hazard mask (baseline Canny)
  D. true positives
  E. false positives
  F. false negatives
  G. overlay

Also decomposes the reference mask into its constituent signals to identify
which signals Canny misses most.

Uses ONLY val_files (200 images). Test set is never touched.
"""
import os
import sys
import json
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.model import LunarUNet

CHECKPOINT = "weights/lunar_unet_v2_best.pth"
SPLIT_JSON = "data/processed_pairs/split.json"
HR_DIR = "data/processed_pairs/hr"
LR_DIR = "data/processed_pairs/lr"
OUT_DIR = "experiments/hazard_baseline"
VIS_DIR = "experiments/visualizations"


def load_model(ckpt, device):
    m = LunarUNet(1, 1).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    m.eval()
    return m


def infer(model, lr_img, device):
    t = torch.from_numpy(lr_img).unsqueeze(0).unsqueeze(0).float() / 255.0
    with torch.inference_mode():
        out = model(t.to(device))
    return (out.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)


def build_ref_mask(hr):
    k_top = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_grd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_saf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh, _ = cv2.threshold(hr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    shadow = hr < max(25, thresh * 0.5)
    top_hat = cv2.morphologyEx(hr, cv2.MORPH_TOPHAT, k_top)
    black_hat = cv2.morphologyEx(hr, cv2.MORPH_BLACKHAT, k_top)
    morph_grad = cv2.morphologyEx(hr, cv2.MORPH_GRADIENT, k_grd)
    raw = shadow | (top_hat > 25) | (black_hat > 25) | (morph_grad > 35)
    dilated = cv2.dilate(raw.astype(np.uint8), k_saf, iterations=1)
    return (dilated > 0).astype(np.uint8), shadow, (top_hat > 25), (black_hat > 25), (morph_grad > 35)


def baseline_hazard(pred):
    edges = cv2.Canny(pred, 80, 200)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    d = cv2.dilate(edges, k, iterations=2)
    return (d > 0).astype(np.uint8)


def overlay_mask(img_bgr, mask, color, alpha=0.4):
    overlay = img_bgr.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)


def save_fn_panel(path, lr, hr, pred, ref_mask, pred_mask,
                  shadow_comp, tophat_comp, blackhat_comp, grad_comp, fname):
    h, w = hr.shape
    bg = np.full((h, w, 3), 30, dtype=np.uint8)

    def g(arr):
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    tp_mask = ((pred_mask > 0) & (ref_mask > 0)).astype(np.uint8)
    fp_mask = ((pred_mask > 0) & (ref_mask == 0)).astype(np.uint8)
    fn_mask = ((pred_mask == 0) & (ref_mask > 0)).astype(np.uint8)

    # Overlay panel: HR with TP=green, FP=red, FN=blue
    ovl = g(hr).copy()
    ovl[tp_mask > 0] = [0, 200, 0]
    ovl[fp_mask > 0] = [0, 0, 200]
    ovl[fn_mask > 0] = [200, 0, 0]

    # FN decomposition: which reference component are they?
    fn_pixels = (pred_mask == 0) & (ref_mask > 0)
    fn_shadow = fn_pixels & shadow_comp
    fn_tophat = fn_pixels & tophat_comp & ~shadow_comp
    fn_bhat = fn_pixels & blackhat_comp & ~shadow_comp & ~tophat_comp
    fn_grad = fn_pixels & grad_comp & ~shadow_comp & ~tophat_comp & ~blackhat_comp

    # Component decomp panel
    decomp = g(hr).copy()
    decomp[fn_shadow] = [255, 100, 0]    # orange = shadow FN
    decomp[fn_tophat] = [0, 255, 200]    # cyan = tophat FN
    decomp[fn_bhat] = [200, 0, 255]      # purple = blackhat FN
    decomp[fn_grad] = [0, 200, 255]      # yellow = grad FN

    labels = [
        (g(lr), "LR Input"),
        (g(pred), "Restored (ep57)"),
        (g(hr), "HR Reference"),
        (overlay_mask(g(hr), ref_mask, [0, 200, 200]), "Ref Hazard (derived)"),
        (overlay_mask(g(pred), pred_mask, [0, 200, 200]), "Pred Hazard (Canny)"),
        (ovl, "TP=grn FP=red FN=blu"),
        (decomp, "FN: shad=org th=cyn bh=pur gr=yel"),
    ]

    panels = []
    for img, label in labels:
        p = img.copy()
        cv2.rectangle(p, (0, 0), (w, 18), (0, 0, 0), -1)
        cv2.putText(p, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        panels.append(p)

    row = np.hstack(panels)

    # Stats header
    total_ref = ref_mask.sum()
    tp_n = tp_mask.sum(); fp_n = fp_mask.sum(); fn_n = fn_mask.sum()
    prec = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0
    rec = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    iou = tp_n / (tp_n + fp_n + fn_n) if (tp_n + fp_n + fn_n) > 0 else 0

    fn_shad_pct = 100 * fn_shadow.sum() / max(fn_n, 1)
    fn_th_pct = 100 * fn_tophat.sum() / max(fn_n, 1)
    fn_bh_pct = 100 * fn_bhat.sum() / max(fn_n, 1)
    fn_gr_pct = 100 * fn_grad.sum() / max(fn_n, 1)

    hdr = np.zeros((40, row.shape[1], 3), dtype=np.uint8)
    txt = (f"{fname}  |  P={prec:.2f} R={rec:.2f} F1={f1:.2f} IoU={iou:.2f}  |  "
           f"FN: shadow={fn_shad_pct:.0f}% tophat={fn_th_pct:.0f}% blackhat={fn_bh_pct:.0f}% grad={fn_gr_pct:.0f}%")
    cv2.putText(hdr, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
    final = np.vstack([hdr, row])
    cv2.imwrite(path, final)

    return {
        "fn_shadow_pct": fn_shad_pct,
        "fn_tophat_pct": fn_th_pct,
        "fn_blackhat_pct": fn_bh_pct,
        "fn_grad_pct": fn_gr_pct,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "iou": iou,
        "total_ref_pixels": int(total_ref),
        "fn_total": int(fn_n),
        "fp_total": int(fp_n),
        "tp_total": int(tp_n),
    }


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    with open(SPLIT_JSON) as f:
        split = json.load(f)
    val_files = split["val_files"]

    model = load_model(CHECKPOINT, device)
    print(f"Model loaded. Val files: {len(val_files)}")

    # Aggregate across all 200 val images
    total_tp = total_tn = total_fp = total_fn = 0
    fn_shadow_acc = fn_tophat_acc = fn_blackhat_acc = fn_grad_acc = 0
    sample_stats = []
    per_image = []

    for idx, fname in enumerate(val_files):
        hr = cv2.imread(os.path.join(HR_DIR, fname), cv2.IMREAD_GRAYSCALE)
        lr = cv2.imread(os.path.join(LR_DIR, fname), cv2.IMREAD_GRAYSCALE)
        if hr is None or lr is None:
            continue

        pred = infer(model, lr, device)
        ref_mask, shadow_c, tophat_c, blackhat_c, grad_c = build_ref_mask(hr)
        pred_mask = baseline_hazard(pred)

        tp = int(((pred_mask > 0) & (ref_mask > 0)).sum())
        tn = int(((pred_mask == 0) & (ref_mask == 0)).sum())
        fp = int(((pred_mask > 0) & (ref_mask == 0)).sum())
        fn = int(((pred_mask == 0) & (ref_mask > 0)).sum())
        total_tp += tp; total_tn += tn; total_fp += fp; total_fn += fn

        fn_pixels = (pred_mask == 0) & (ref_mask > 0)
        fn_shadow_acc += int((fn_pixels & shadow_c).sum())
        fn_tophat_acc += int((fn_pixels & tophat_c & ~shadow_c).sum())
        fn_blackhat_acc += int((fn_pixels & blackhat_c & ~shadow_c & ~tophat_c).sum())
        fn_grad_acc += int((fn_pixels & grad_c & ~shadow_c & ~tophat_c & ~blackhat_c).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        per_image.append({"fname": fname, "f1": f1, "iou": iou, "recall": rec, "precision": prec,
                           "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                           "pred": pred, "ref_mask": ref_mask, "pred_mask": pred_mask,
                           "hr": hr, "lr": lr,
                           "shadow_c": shadow_c, "tophat_c": tophat_c,
                           "blackhat_c": blackhat_c, "grad_c": grad_c})

        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(val_files)} processed...")

    # Aggregate metrics
    total = total_tp + total_tn + total_fp + total_fn
    acc = (total_tp + total_tn) / total
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0

    total_fn_all = total_fn
    fn_shadow_pct = 100 * fn_shadow_acc / max(total_fn_all, 1)
    fn_tophat_pct = 100 * fn_tophat_acc / max(total_fn_all, 1)
    fn_bhat_pct = 100 * fn_blackhat_acc / max(total_fn_all, 1)
    fn_grad_pct = 100 * fn_grad_acc / max(total_fn_all, 1)

    print("\n=== BASELINE METRICS (Val set, 200 images) ===")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1:        {f1*100:.2f}%")
    print(f"  IoU:       {iou*100:.2f}%")
    print(f"  TP={total_tp:,} TN={total_tn:,} FP={total_fp:,} FN={total_fn_all:,}")
    print(f"\n=== FALSE NEGATIVE DECOMPOSITION ===")
    print(f"  Shadow component:    {fn_shadow_pct:.1f}% of all FN pixels")
    print(f"  Top-hat component:   {fn_tophat_pct:.1f}%")
    print(f"  Black-hat component: {fn_bhat_pct:.1f}%")
    print(f"  Morph-grad component:{fn_grad_pct:.1f}%")
    print(f"  --> Root cause: FN pixels are dominated by the signal NOT detected by Canny")

    # Save visualizations for representative samples: worst recall, median, best recall
    per_image.sort(key=lambda x: x["recall"])
    vis_candidates = []
    if len(per_image) >= 3:
        vis_candidates.append(per_image[0])           # worst recall
        vis_candidates.append(per_image[len(per_image)//2])  # median
        vis_candidates.append(per_image[-1])           # best recall
        vis_candidates.append(per_image[len(per_image)//4])
        vis_candidates.append(per_image[3*len(per_image)//4])

    print(f"\nSaving {len(vis_candidates)} representative FN analysis panels...")
    for i, s in enumerate(vis_candidates):
        tag = ["worst_recall", "q25", "median", "q75", "best_recall"][i]
        out_path = os.path.join(VIS_DIR, f"fn_analysis_{tag}_{s['fname']}")
        save_fn_panel(out_path, s["lr"], s["hr"], s["pred"], s["ref_mask"], s["pred_mask"],
                      s["shadow_c"], s["tophat_c"], s["blackhat_c"], s["grad_c"], s["fname"])
        print(f"  Saved: {out_path}")

    # Save baseline results JSON
    baseline_result = {
        "checkpoint": CHECKPOINT,
        "split": "validation",
        "n_images": len(per_image),
        "method": "canny_baseline",
        "canny_low": 80, "canny_high": 200,
        "kernel": 11, "iterations": 2,
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1 * 100, 2),
        "iou": round(iou * 100, 2),
        "tp": total_tp, "tn": total_tn, "fp": total_fp, "fn": total_fn_all,
        "fn_decomposition": {
            "shadow_pct": round(fn_shadow_pct, 1),
            "tophat_pct": round(fn_tophat_pct, 1),
            "blackhat_pct": round(fn_bhat_pct, 1),
            "morph_grad_pct": round(fn_grad_pct, 1),
            "shadow_pixels": fn_shadow_acc,
            "tophat_pixels": fn_tophat_acc,
            "blackhat_pixels": fn_blackhat_acc,
            "morph_grad_pixels": fn_grad_acc,
        },
        "diagnosis": (
            "The baseline Canny detector misses dark shadow/crater-interior pixels because "
            "Canny detects EDGES (intensity transitions) not homogeneous dark regions. "
            "Shadow regions that are uniformly dark have no strong edge response and are "
            "entirely missed. Top-hat (bright protrusions) and black-hat (dark depressions) "
            "are partially captured at their boundaries but interiors are missed. "
            "Adding explicit dark-region detection should dramatically improve recall."
        )
    }

    out_json = os.path.join(OUT_DIR, "baseline_val_metrics.json")
    with open(out_json, "w") as f:
        json.dump(baseline_result, f, indent=2)
    print(f"\nBaseline metrics saved: {out_json}")

    return baseline_result


if __name__ == "__main__":
    main()
