"""
Hazard V2 Optimizer — second-generation precision/recall optimization.

Key architectural changes from V1:
  1. Continuous score map — no hard dilation before thresholding
  2. Proper feature normalization (all features to [0,1])
  3. Hazard CORE mask separated from safety buffer
  4. Full P-R curve via threshold sweep on continuous score
  5. Component size filtering as a separate tunable stage
  6. Multi-scale feature extraction
  7. Local/adaptive shadow detection
  8. Pareto frontier identification

Optimization metric priority: F1, then IoU (not Accuracy).
Uses ONLY validation set (200 images). Test set NEVER touched here.

Reference: DERIVED PHOTOMETRIC REFERENCE (not human ground truth).
"""

import os, sys, json, time, csv, cv2, numpy as np, torch
from itertools import product
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.model import LunarUNet

CKPT = "weights/lunar_unet_v2_best.pth"
HR_DIR = "data/processed_pairs/hr"
LR_DIR = "data/processed_pairs/lr"
OUT_DIR = "evaluation_results/hazard_v2"
VIS_DIR = "experiments/visualizations"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_model(ckpt, device):
    m = LunarUNet(1, 1).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    m.eval()
    return m


def infer(model, lr, device):
    t = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).float() / 255.
    with torch.inference_mode():
        return (model(t.to(device)).squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def build_ref_mask(hr):
    k_t = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_g = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thr, _ = cv2.threshold(hr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    shd = hr < max(25, thr * 0.5)
    th = cv2.morphologyEx(hr, cv2.MORPH_TOPHAT, k_t)
    bh = cv2.morphologyEx(hr, cv2.MORPH_BLACKHAT, k_t)
    mg = cv2.morphologyEx(hr, cv2.MORPH_GRADIENT, k_g)
    raw = shd | (th > 25) | (bh > 25) | (mg > 35)
    return (cv2.dilate(raw.astype(np.uint8), k_s, iterations=1) > 0).astype(np.uint8)


def hm(tp, tn, fp, fn):
    t = tp + tn + fp + fn
    acc = (tp + tn) / t if t > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, iou=iou,
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


def apply_component_filter(mask, min_area):
    if min_area <= 1:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out


# ── feature extraction ────────────────────────────────────────────────────────

def extract_features(pred):
    """
    Extract normalised [0,1] feature maps from a restored image.
    Returns dict of feature maps, all float32, same shape as pred.
    """
    h, w = pred.shape
    f = {}

    # 1. Raw Canny edges — multiple threshold pairs, NO dilation
    for cl, ch in [(20, 60), (30, 80), (40, 100), (50, 120), (60, 150)]:
        e = cv2.Canny(pred, cl, ch).astype(np.float32) / 255.
        f[f'canny_{cl}_{ch}'] = e

    # 2. Canny with MINIMAL dilation (3×3, 1 iter) — thin safety margin only
    for cl, ch in [(30, 80), (50, 120)]:
        e = cv2.Canny(pred, cl, ch)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ed = cv2.dilate(e, k3, iterations=1).astype(np.float32) / 255.
        f[f'canny_dil3_{cl}_{ch}'] = ed

    # 3. Sobel gradient magnitude, normalised to [0,1]
    for ks in [3, 5]:
        sx = cv2.Sobel(pred.astype(np.float32), cv2.CV_32F, 1, 0, ksize=ks)
        sy = cv2.Sobel(pred.astype(np.float32), cv2.CV_32F, 0, 1, ksize=ks)
        mag = np.sqrt(sx ** 2 + sy ** 2)
        p99 = np.percentile(mag, 99)
        f[f'sobel_k{ks}'] = np.clip(mag / (p99 + 1e-6), 0, 1).astype(np.float32)

    # 4. Laplacian (abs), normalised
    lap = np.abs(cv2.Laplacian(pred, cv2.CV_64F)).astype(np.float32)
    p99 = np.percentile(lap, 99)
    f['laplacian'] = np.clip(lap / (p99 + 1e-6), 0, 1).astype(np.float32)

    # 5. Global shadow — multiple thresholds
    for thr in [30, 40, 50, 60]:
        f[f'shadow_{thr}'] = (pred < thr).astype(np.float32)

    # 6. Local/adaptive shadow — pixel below local_mean - k*local_std
    pred_f = pred.astype(np.float32)
    for block in [31, 51]:
        local_mean = cv2.GaussianBlur(pred_f, (block, block), 0)
        diff = local_mean - pred_f  # positive where pixel is darker than local mean
        p90 = max(np.percentile(diff[diff > 0], 90) if (diff > 0).any() else 1.0, 1.0)
        f[f'local_dark_{block}'] = np.clip(diff / p90, 0, 1).astype(np.float32)

    # 7. TopHat / BlackHat at multiple scales
    for sz in [7, 11, 15]:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))
        th = cv2.morphologyEx(pred, cv2.MORPH_TOPHAT, k).astype(np.float32)
        bh = cv2.morphologyEx(pred, cv2.MORPH_BLACKHAT, k).astype(np.float32)
        mg = cv2.morphologyEx(pred, cv2.MORPH_GRADIENT, k).astype(np.float32)
        # normalise each
        for name, arr in [(f'tophat_{sz}', th), (f'blackhat_{sz}', bh), (f'morph_grad_{sz}', mg)]:
            p99 = max(np.percentile(arr, 99), 1.0)
            f[name] = np.clip(arr / p99, 0, 1).astype(np.float32)

    return f


def score_map(features, weights):
    """Weighted sum of selected features. Weights dict: feature_name → float."""
    total_w = sum(abs(v) for v in weights.values())
    if total_w < 1e-9:
        return np.zeros_like(next(iter(features.values())))
    score = np.zeros_like(next(iter(features.values())), dtype=np.float32)
    for k, w in weights.items():
        if k in features:
            score += w * features[k]
    return score / total_w  # normalise to [0,1] range


def mask_from_score(score, threshold, min_component=0):
    binary = (score >= threshold).astype(np.uint8)
    if min_component > 1:
        binary = apply_component_filter(binary, min_component)
    return binary


# ── metric aggregation ────────────────────────────────────────────────────────

def eval_threshold_on_samples(sample_scores, ref_masks, threshold, min_component=0):
    tp = tn = fp = fn = 0
    for score, rm in zip(sample_scores, ref_masks):
        pm = mask_from_score(score, threshold, min_component)
        tp += int(((pm > 0) & (rm > 0)).sum())
        tn += int(((pm == 0) & (rm == 0)).sum())
        fp += int(((pm > 0) & (rm == 0)).sum())
        fn += int(((pm == 0) & (rm > 0)).sum())
    return hm(tp, tn, fp, fn)


def pr_curve(sample_scores, ref_masks, thresholds, min_component=0):
    rows = []
    for thr in thresholds:
        m = eval_threshold_on_samples(sample_scores, ref_masks, thr, min_component)
        m['threshold'] = thr
        pred_px = sum(int((s >= thr).sum()) for s in sample_scores)
        total_px = sum(rm.size for rm in ref_masks)
        m['pred_hazard_frac'] = pred_px / total_px
        rows.append(m)
    return rows


def find_pareto(rows):
    """Find Pareto-optimal (precision, recall) points."""
    pareto = []
    for r in sorted(rows, key=lambda x: -x['recall']):
        if not pareto or r['precision'] > pareto[-1]['precision']:
            pareto.append(r)
    return pareto


# ── weight configurations ─────────────────────────────────────────────────────

def make_weight_configs():
    """
    Return named weight configurations for the score map.
    Keys are feature names from extract_features().
    Values are relative weights (will be normalised to sum to 1).
    """
    configs = {}

    # Shadow-heavy: prioritises dark regions (high recall, moderate precision)
    configs['shadow_heavy'] = {
        'shadow_40': 0.40, 'local_dark_31': 0.25,
        'canny_dil3_30_80': 0.10, 'sobel_k3': 0.10,
        'blackhat_11': 0.15,
    }

    # Edge-heavy: prioritises gradient/edge signals
    configs['edge_heavy'] = {
        'canny_dil3_30_80': 0.25, 'canny_dil3_50_120': 0.20,
        'sobel_k3': 0.25, 'laplacian': 0.15,
        'morph_grad_7': 0.15,
    }

    # Balanced: roughly equal contribution
    configs['balanced'] = {
        'shadow_40': 0.20, 'local_dark_31': 0.10,
        'canny_dil3_30_80': 0.15, 'sobel_k3': 0.15,
        'laplacian': 0.10, 'blackhat_11': 0.15,
        'tophat_11': 0.15,
    }

    # Morphology-heavy: top-hat / black-hat / morph-gradient
    configs['morphology_heavy'] = {
        'tophat_7': 0.15, 'tophat_11': 0.15,
        'blackhat_7': 0.15, 'blackhat_11': 0.15,
        'morph_grad_7': 0.20, 'morph_grad_11': 0.20,
    }

    # Conservative (precision-oriented): high-confidence signals only
    configs['conservative'] = {
        'shadow_30': 0.30,
        'canny_20_60': 0.20,
        'blackhat_11': 0.25,
        'morph_grad_7': 0.25,
    }

    # Local-adaptive: prefers local-normalization-based features
    configs['local_adaptive'] = {
        'local_dark_31': 0.30, 'local_dark_51': 0.20,
        'canny_dil3_30_80': 0.20, 'sobel_k3': 0.15,
        'blackhat_11': 0.15,
    }

    # Shadow + morph (no raw canny dilation)
    configs['shadow_morph'] = {
        'shadow_40': 0.25, 'local_dark_31': 0.20,
        'blackhat_11': 0.20, 'morph_grad_7': 0.20,
        'tophat_11': 0.15,
    }

    # Tuned balanced — based on FP analysis: reduce canny weight, boost shadow/morph
    configs['tuned_balanced'] = {
        'shadow_40': 0.22, 'local_dark_31': 0.18,
        'canny_dil3_30_80': 0.12, 'sobel_k3': 0.13,
        'laplacian': 0.08, 'blackhat_11': 0.14,
        'morph_grad_7': 0.13,
    }

    # Multi-scale balanced
    configs['multiscale_balanced'] = {
        'shadow_40': 0.18, 'local_dark_31': 0.12,
        'canny_dil3_30_80': 0.10, 'canny_dil3_50_120': 0.08,
        'sobel_k3': 0.10, 'laplacian': 0.07,
        'blackhat_7': 0.10, 'blackhat_11': 0.10,
        'morph_grad_7': 0.08, 'morph_grad_11': 0.07,
    }

    # Precision-biased high-recall
    configs['precision_biased'] = {
        'shadow_30': 0.25, 'canny_20_60': 0.15,
        'blackhat_11': 0.20, 'morph_grad_7': 0.15,
        'local_dark_31': 0.25,
    }

    return configs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    model = load_model(CKPT, device)

    with open("data/processed_pairs/split.json") as f:
        val_files = json.load(f)["val_files"]
    print(f"Val images: {len(val_files)}")

    # ── Phase 0: Pre-compute all feature maps and ref masks (GPU inference once) ──
    print("\nPhase 0: Pre-computing feature maps and ref masks...")
    all_features = []
    all_ref_masks = []
    total_ref_px = 0
    total_px = 0

    for i, fname in enumerate(val_files):
        hr = cv2.imread(os.path.join(HR_DIR, fname), cv2.IMREAD_GRAYSCALE)
        lr = cv2.imread(os.path.join(LR_DIR, fname), cv2.IMREAD_GRAYSCALE)
        if hr is None or lr is None:
            continue
        pred = infer(model, lr, device)
        feats = extract_features(pred)
        rm = build_ref_mask(hr)
        all_features.append(feats)
        all_ref_masks.append(rm)
        total_ref_px += int((rm > 0).sum())
        total_px += rm.size
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(val_files)} done...")

    ref_hazard_frac = total_ref_px / total_px
    print(f"  Pre-computed {len(all_features)} samples")
    print(f"  Reference hazard fraction: {ref_hazard_frac*100:.1f}%")

    # ── Phase 1: V1 improved baseline measurement (for comparison) ──
    print("\nPhase 1: Measuring V1 improved baseline on val set...")
    with open("evaluation_results/hazard_optimization.json") as f:
        opt = json.load(f)
    v1_cfg = opt["overall_best"]
    print(f"  V1 best: method={v1_cfg['method']} F1={v1_cfg['f1']*100:.2f}% IoU={v1_cfg['iou']*100:.2f}%")
    print(f"  V1 Precision={v1_cfg['precision']*100:.2f}% Recall={v1_cfg['recall']*100:.2f}%")

    # ── Phase 2: Threshold sweep on each weight configuration ──
    print("\nPhase 2: Threshold sweep across weight configurations...")
    THRESHOLDS = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225,
                  0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40, 0.45,
                  0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    MIN_COMPONENTS = [0, 5, 20, 50]  # component size filter sweep

    weight_configs = make_weight_configs()
    all_results = []
    best_f1 = {"f1": 0}
    best_iou = {"iou": 0}
    best_precision_r80 = {"precision": 0}  # max precision with recall >= 0.80
    best_f1_r80 = {"f1": 0}                # max F1 with recall >= 0.80
    best_iou_r80 = {"iou": 0}              # max IoU with recall >= 0.80
    best_f1_r85 = {"f1": 0}
    best_f1_r90 = {"f1": 0}

    total_runs = len(weight_configs) * len(THRESHOLDS) * len(MIN_COMPONENTS)
    run_idx = 0

    for wname, weights in weight_configs.items():
        # Build score maps for all val images with this weight config
        sample_scores = [score_map(f, weights) for f in all_features]

        for min_comp in MIN_COMPONENTS:
            curve = pr_curve(sample_scores, all_ref_masks, THRESHOLDS, min_comp)

            for row in curve:
                run_idx += 1
                entry = {
                    "weight_config": wname,
                    "min_component": min_comp,
                    "ref_hazard_frac": round(ref_hazard_frac, 4),
                    **row,
                }
                all_results.append(entry)

                m = row
                if m["f1"] > best_f1.get("f1", 0):
                    best_f1 = entry.copy()
                if m["iou"] > best_iou.get("iou", 0):
                    best_iou = entry.copy()
                if m["recall"] >= 0.80:
                    if m["precision"] > best_precision_r80.get("precision", 0):
                        best_precision_r80 = entry.copy()
                    if m["f1"] > best_f1_r80.get("f1", 0):
                        best_f1_r80 = entry.copy()
                    if m["iou"] > best_iou_r80.get("iou", 0):
                        best_iou_r80 = entry.copy()
                if m["recall"] >= 0.85 and m["f1"] > best_f1_r85.get("f1", 0):
                    best_f1_r85 = entry.copy()
                if m["recall"] >= 0.90 and m["f1"] > best_f1_r90.get("f1", 0):
                    best_f1_r90 = entry.copy()

        if run_idx % (len(THRESHOLDS) * len(MIN_COMPONENTS) * 2) == 0:
            print(f"  [{run_idx}/{total_runs}] best F1 so far: {best_f1.get('f1',0)*100:.2f}%")

    print(f"\nCompleted {len(all_results)} configurations")

    # ── Phase 3: Component sweep (dedicated, using best weight config) ──
    print("\nPhase 3: Component size sweep on best config...")
    best_wname = best_f1_r80.get("weight_config", best_f1.get("weight_config", "balanced"))
    best_thr = best_f1_r80.get("threshold", 0.30)
    best_scores = [score_map(f, weight_configs[best_wname]) for f in all_features]
    comp_results = []
    for min_comp in [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        m = eval_threshold_on_samples(best_scores, all_ref_masks, best_thr, min_comp)
        pred_px = sum(int(mask_from_score(s, best_thr, min_comp).sum()) for s in best_scores)
        m["min_component"] = min_comp
        m["weight_config"] = best_wname
        m["threshold"] = best_thr
        m["pred_hazard_frac"] = pred_px / total_px
        comp_results.append(m)
        print(f"  min_comp={min_comp:5d}: P={m['precision']*100:.1f}% R={m['recall']*100:.1f}% "
              f"F1={m['f1']*100:.1f}% IoU={m['iou']*100:.1f}% pred_frac={m['pred_hazard_frac']*100:.1f}%")

    # ── Phase 4: Save CSVs and JSON ──
    print("\nPhase 4: Saving results...")

    # threshold_sweep.csv — all results
    fields = ["weight_config", "threshold", "min_component", "accuracy", "precision",
              "recall", "f1", "iou", "tp", "tn", "fp", "fn",
              "pred_hazard_frac", "ref_hazard_frac"]
    with open(os.path.join(OUT_DIR, "threshold_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(all_results, key=lambda x: -x["f1"]):
            w.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()})

    # component_sweep.csv
    with open(os.path.join(OUT_DIR, "component_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["min_component", "weight_config", "threshold",
                                          "precision", "recall", "f1", "iou", "pred_hazard_frac",
                                          "tp", "tn", "fp", "fn"], extrasaction="ignore")
        w.writeheader()
        for r in comp_results:
            w.writerow(r)

    # precision_recall_curve.csv — best weight config only
    pr_rows = [r for r in all_results
               if r["weight_config"] == best_wname and r["min_component"] == 0]
    pr_rows.sort(key=lambda x: x["recall"])
    with open(os.path.join(OUT_DIR, "precision_recall_curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["threshold", "precision", "recall", "f1", "iou",
                                          "pred_hazard_frac"], extrasaction="ignore")
        try:
            w.writeheader()
            for r in pr_rows:
                w.writerow({k: round(r[k], 6) for k in ["threshold","precision","recall","f1","iou","pred_hazard_frac"]})
        except Exception:
            pass

    # pareto_frontier.csv
    pareto = find_pareto(pr_rows)
    with open(os.path.join(OUT_DIR, "pareto_frontier.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["threshold", "precision", "recall", "f1", "iou",
                                          "pred_hazard_frac"], extrasaction="ignore")
        w.writeheader()
        for r in pareto:
            w.writerow({k: round(r.get(k, 0), 6) for k in
                        ["threshold","precision","recall","f1","iou","pred_hazard_frac"]})

    # Print tradeoff table
    print("\n" + "=" * 80)
    print("TRADEOFF TABLE — Best config per weight_config, sorted by F1")
    print("=" * 80)
    print(f"{'Config':<22} {'Thr':>5} {'Comp':>5} {'P%':>7} {'R%':>7} {'F1%':>7} {'IoU%':>7} {'PredFrac%':>10}")
    print("-" * 80)

    # Best per weight_config
    best_per_config = {}
    for r in all_results:
        wn = r["weight_config"]
        if wn not in best_per_config or r["f1"] > best_per_config[wn]["f1"]:
            best_per_config[wn] = r
    for r in sorted(best_per_config.values(), key=lambda x: -x["f1"]):
        print(f"{r['weight_config']:<22} {r['threshold']:>5.3f} {r['min_component']:>5} "
              f"{r['precision']*100:>7.2f} {r['recall']*100:>7.2f} {r['f1']*100:>7.2f} "
              f"{r['iou']*100:>7.2f} {r['pred_hazard_frac']*100:>10.1f}%")

    # ── Phase 5: Select operating points ──
    print("\n" + "=" * 80)
    print("OPERATING POINTS (selected using VALIDATION ONLY)")
    print("=" * 80)
    print(f"Reference hazard fraction: {ref_hazard_frac*100:.1f}%")
    print()

    ops = {
        "CONSERVATIVE (max recall)":   max(all_results, key=lambda x: (x["recall"], x["f1"])),
        "BALANCED (max F1)":            best_f1,
        "BALANCED_IoU (max IoU)":       best_iou,
        "PRECISION_R80 (max P, R≥0.80)": best_precision_r80 if best_precision_r80.get("precision", 0) > 0 else {"note": "No config meets R≥0.80"},
        "BEST_F1_R80 (max F1, R≥0.80)": best_f1_r80 if best_f1_r80.get("f1", 0) > 0 else {"note": "No config meets R≥0.80"},
        "BEST_F1_R85 (max F1, R≥0.85)": best_f1_r85 if best_f1_r85.get("f1", 0) > 0 else {"note": "No config meets R≥0.85"},
        "BEST_F1_R90 (max F1, R≥0.90)": best_f1_r90 if best_f1_r90.get("f1", 0) > 0 else {"note": "No config meets R≥0.90"},
    }

    for label, op in ops.items():
        if "note" in op:
            print(f"{label}: {op['note']}")
            continue
        print(f"{label}:")
        print(f"  Config: {op.get('weight_config','?')} thr={op.get('threshold','?'):.3f} "
              f"min_comp={op.get('min_component',0)}")
        print(f"  P={op.get('precision',0)*100:.2f}% R={op.get('recall',0)*100:.2f}% "
              f"F1={op.get('f1',0)*100:.2f}% IoU={op.get('iou',0)*100:.2f}%")
        print(f"  PredFrac={op.get('pred_hazard_frac',0)*100:.1f}% | "
              f"TP={op.get('tp',0):,} FP={op.get('fp',0):,} FN={op.get('fn',0):,}")
        print()

    # ── Phase 6: Save best configs JSON ──
    best_configs = {
        "experiment": "hazard_v2",
        "checkpoint": CKPT,
        "val_count": len(all_features),
        "ref_hazard_frac": round(ref_hazard_frac, 4),
        "v1_baseline_val": {
            "precision": v1_cfg["precision"], "recall": v1_cfg["recall"],
            "f1": v1_cfg["f1"], "iou": v1_cfg["iou"]
        },
        "operating_points": {}
    }

    op_keys = {
        "conservative": "CONSERVATIVE (max recall)",
        "balanced_f1": "BALANCED (max F1)",
        "balanced_iou": "BALANCED_IoU (max IoU)",
        "precision_r80": "PRECISION_R80 (max P, R≥0.80)",
        "best_f1_r80": "BEST_F1_R80 (max F1, R≥0.80)",
        "best_f1_r85": "BEST_F1_R85 (max F1, R≥0.85)",
        "best_f1_r90": "BEST_F1_R90 (max F1, R≥0.90)",
    }
    for key, label in op_keys.items():
        op = ops[label]
        if "note" in op:
            best_configs["operating_points"][key] = {"available": False, "note": op["note"]}
        else:
            best_configs["operating_points"][key] = {
                "available": True,
                "weight_config": op.get("weight_config"),
                "weights": weight_configs.get(op.get("weight_config", ""), {}),
                "threshold": op.get("threshold"),
                "min_component": op.get("min_component", 0),
                "validation_metrics": {
                    "precision": round(op.get("precision", 0), 4),
                    "recall": round(op.get("recall", 0), 4),
                    "f1": round(op.get("f1", 0), 4),
                    "iou": round(op.get("iou", 0), 4),
                    "accuracy": round(op.get("accuracy", 0), 4),
                    "tp": op.get("tp", 0),
                    "tn": op.get("tn", 0),
                    "fp": op.get("fp", 0),
                    "fn": op.get("fn", 0),
                    "pred_hazard_frac": round(op.get("pred_hazard_frac", 0), 4),
                    "ref_hazard_frac": round(ref_hazard_frac, 4),
                }
            }

    with open(os.path.join(OUT_DIR, "best_validation_configs.json"), "w") as f:
        json.dump(best_configs, f, indent=2)

    print(f"\nResults saved to {OUT_DIR}/")
    print(f"  threshold_sweep.csv")
    print(f"  component_sweep.csv")
    print(f"  precision_recall_curve.csv")
    print(f"  pareto_frontier.csv")
    print(f"  best_validation_configs.json")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    return best_configs


if __name__ == "__main__":
    main()
