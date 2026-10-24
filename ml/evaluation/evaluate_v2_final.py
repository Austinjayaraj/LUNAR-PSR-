"""
V2 Final evaluation on the untouched 300-image test set.

Compares:
  BASELINE (v1 model, Canny(80,200)+dilation(11x11,2))
  V1_IMPROVED (epoch-57, multi-feature detector from optimize_hazard.py)
  V2_BALANCED (epoch-57, oracle-core + multiscale(7+11) + buf(7x7,2) + mc=50)
  V2_CONSERVATIVE (epoch-57, oracle-core + multiscale(7+15) + buf(7x7,2) + mc=100)
  V2_PRECISION (epoch-57, oracle-core + buf(9x9,2) + mc=100)

ALL CONFIGURATIONS were selected using ONLY validation data.
Test set is untouched until this script runs.

Reference: DERIVED PHOTOMETRIC REFERENCE (not human ground truth).
"""
import os, sys, json, csv, time, cv2, numpy as np, torch
sys.path.insert(0, '.')
from src.model import LunarUNet
from src.metrics import calculate_restoration_metrics, calculate_hazard_metrics

OUT_DIR = "evaluation_results/v2_final"
VIS_DIR = os.path.join(OUT_DIR, "visual_analysis")


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


# ── hazard functions ──────────────────────────────────────────────────────────

def hazard_baseline(pred):
    """Original v1: Canny(80,200)+MORPH_ELLIPSE(11x11) iter=2."""
    edges = cv2.Canny(pred, 80, 200)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    return (cv2.dilate(edges, k, iterations=2) > 0).astype(np.uint8)


def hazard_v1_improved(pred):
    """V1 multi-feature from optimize_hazard.py (best val config)."""
    with open("evaluation_results/hazard_optimization.json") as f:
        cfg = json.load(f)["overall_best"]
    cl, ch, ks, it = cfg["canny_low"], cfg["canny_high"], cfg["kernel"], cfg["iter"]
    edges = cv2.Canny(pred, cl, ch)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    cmap = cv2.dilate(edges, k, iterations=it).astype(np.float32) / 255.
    sx = cv2.Sobel(pred, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(pred, cv2.CV_64F, 0, 1, ksize=3)
    smap = (np.sqrt(sx**2 + sy**2) > cfg["sobel_thresh"]).astype(np.float32)
    lmap = (np.abs(cv2.Laplacian(pred, cv2.CV_64F)) > cfg["lap_thresh"]).astype(np.float32)
    shadow_map = (pred < cfg["shadow_thresh"]).astype(np.float32)
    k_th = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(pred, cv2.MORPH_TOPHAT, k_th)
    blackhat = cv2.morphologyEx(pred, cv2.MORPH_BLACKHAT, k_th)
    thmap = ((tophat > cfg["tophat_thresh"]) | (blackhat > cfg["tophat_thresh"])).astype(np.float32)
    combined = (cfg["w_canny"]*cmap + cfg["w_sobel"]*smap + cfg["w_lap"]*lmap +
                cfg["w_shadow"]*shadow_map + cfg["w_tophat"]*thmap)
    hm = (combined >= cfg["score_thresh"]).astype(np.uint8)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(hm, cv2.MORPH_CLOSE, k_close)


def oracle_core_multi(pred, big_sz=11):
    k_t = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_t_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (big_sz, big_sz))
    k_g = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thr, _ = cv2.threshold(pred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    shd = pred < max(25, thr * 0.5)
    th_s = cv2.morphologyEx(pred, cv2.MORPH_TOPHAT, k_t)
    bh_s = cv2.morphologyEx(pred, cv2.MORPH_BLACKHAT, k_t)
    th_l = cv2.morphologyEx(pred, cv2.MORPH_TOPHAT, k_t_big)
    bh_l = cv2.morphologyEx(pred, cv2.MORPH_BLACKHAT, k_t_big)
    mg = cv2.morphologyEx(pred, cv2.MORPH_GRADIENT, k_g)
    return (shd | (th_s > 25) | (bh_s > 25) | (th_l > 20) | (bh_l > 20) | (mg > 35)).astype(np.uint8)


def comp_filter(mask, min_area):
    if min_area <= 1:
        return mask
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lbl == i] = 1
    return out


def hazard_v2_balanced(pred):
    """V2 balanced: oracle-core(7+11)+buf(7x7x2)+mc=50"""
    core = oracle_core_multi(pred, big_sz=11)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dil = (cv2.dilate(core, k, iterations=2) > 0).astype(np.uint8)
    return comp_filter(dil, 50)


def hazard_v2_conservative(pred):
    """V2 conservative: oracle-core(7+15)+buf(7x7x2)+mc=100"""
    core = oracle_core_multi(pred, big_sz=15)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dil = (cv2.dilate(core, k, iterations=2) > 0).astype(np.uint8)
    return comp_filter(dil, 100)


def hazard_v2_precision(pred):
    """V2 precision: oracle-core(7)+buf(9x9x2)+mc=100"""
    core = oracle_core_multi(pred, big_sz=0)  # small_only
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dil = (cv2.dilate(core, k, iterations=2) > 0).astype(np.uint8)
    return comp_filter(dil, 100)


def hazard_v2_precision_real(pred):
    """V2 precision: oracle-core(7) only, no large kernel, buf(9x9x2)+mc=100"""
    k_t = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_g = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thr, _ = cv2.threshold(pred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    shd = pred < max(25, thr * 0.5)
    th = cv2.morphologyEx(pred, cv2.MORPH_TOPHAT, k_t)
    bh = cv2.morphologyEx(pred, cv2.MORPH_BLACKHAT, k_t)
    mg = cv2.morphologyEx(pred, cv2.MORPH_GRADIENT, k_g)
    core = (shd | (th > 25) | (bh > 25) | (mg > 35)).astype(np.uint8)
    k_buf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dil = (cv2.dilate(core, k_buf, iterations=2) > 0).astype(np.uint8)
    return comp_filter(dil, 100)


# ── aggregate helpers ─────────────────────────────────────────────────────────

def agg_hazard(agg):
    tp, tn, fp, fn = agg['tp'], agg['tn'], agg['fp'], agg['fn']
    total = tp + tn + fp + fn
    pred_px = agg['pred_px']
    ref_px = agg['ref_px']
    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    return {
        'accuracy': round(acc * 100, 2),
        'precision': round(prec * 100, 2),
        'recall': round(rec * 100, 2),
        'f1': round(f1 * 100, 2),
        'iou': round(iou * 100, 2),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'pred_hazard_frac': round(pred_px / total * 100, 1),
        'ref_hazard_frac': round(ref_px / total * 100, 1),
    }


def agg_rest(results):
    keys = ['mae', 'rmse', 'r2', 'psnr', 'ssim']
    out = {}
    for k in keys:
        vals = [r[k] for r in results if r[k] != float('inf')]
        out[k] = {'mean': round(float(np.mean(vals)), 4)}
    return out


def save_error_vis(path, fname, lr, hr, pred, ref_mask,
                   base_haz, v1_haz, v2b_haz, v2c_haz, v2p_haz):
    h, w = hr.shape

    def g(arr):
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    def mask_vis(m, color=(0, 200, 200)):
        v = np.zeros((h, w, 3), dtype=np.uint8)
        v[m > 0] = color
        return v

    def fp_fn_vis(pm, rm):
        v = g(hr).copy()
        v[(pm > 0) & (rm == 0)] = [0, 0, 220]   # FP red
        v[(pm == 0) & (rm > 0)] = [220, 0, 0]   # FN blue
        v[(pm > 0) & (rm > 0)] = [0, 180, 0]    # TP green
        return v

    row1 = np.hstack([g(lr), g(hr), g(pred), mask_vis(ref_mask)])
    row2 = np.hstack([fp_fn_vis(base_haz, ref_mask),
                      fp_fn_vis(v1_haz, ref_mask),
                      fp_fn_vis(v2b_haz, ref_mask),
                      fp_fn_vis(v2p_haz, ref_mask)])

    labels_r1 = ["LR Input", "HR Reference", "Restored (ep57)", "Ref Hazard (derived)"]
    labels_r2 = ["Baseline", "V1 Improved", "V2 Balanced", "V2 Precision"]

    for row, labels in [(row1, labels_r1), (row2, labels_r2)]:
        for i, label in enumerate(labels):
            x = i * w + 4
            cv2.rectangle(row, (x, 0), (x + len(label) * 8, 16), (0, 0, 0), -1)
            cv2.putText(row, label, (x + 3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    hdr = np.zeros((24, row1.shape[1], 3), dtype=np.uint8)
    cv2.putText(hdr, f"{fname}  |  green=TP  red=FP  blue=FN", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
    cv2.imwrite(path, np.vstack([hdr, row1, row2]))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    baseline_model = load_model("weights/lunar_unet_best.pth", device)
    improved_model = load_model("weights/lunar_unet_v2_best.pth", device)

    with open("data/processed_pairs/split.json") as f:
        test_files = json.load(f)["test_files"]
    HR_DIR = "data/processed_pairs/hr"
    LR_DIR = "data/processed_pairs/lr"
    print(f"Test set: {len(test_files)} images (UNTOUCHED until now)\n")

    methods = {
        'baseline':       (baseline_model, hazard_baseline),
        'v1_improved':    (improved_model, hazard_v1_improved),
        'v2_balanced':    (improved_model, hazard_v2_balanced),
        'v2_conservative':(improved_model, hazard_v2_conservative),
        'v2_precision':   (improved_model, hazard_v2_precision_real),
    }

    aggs = {name: {'tp':0,'tn':0,'fp':0,'fn':0,'pred_px':0,'ref_px':0}
            for name in methods}
    rest_baseline = []
    rest_improved = []
    all_samples = []

    for idx, fname in enumerate(test_files):
        hr = cv2.imread(os.path.join(HR_DIR, fname), cv2.IMREAD_GRAYSCALE)
        lr = cv2.imread(os.path.join(LR_DIR, fname), cv2.IMREAD_GRAYSCALE)
        if hr is None or lr is None:
            continue

        ref = build_ref_mask(hr)
        base_pred = infer(baseline_model, lr, device)
        imp_pred = infer(improved_model, lr, device)

        rest_baseline.append(calculate_restoration_metrics(base_pred, hr))
        rest_improved.append(calculate_restoration_metrics(imp_pred, hr))

        per_method_haz = {}
        for name, (mdl, fn) in methods.items():
            pred = base_pred if name == 'baseline' else imp_pred
            pm = fn(pred)
            per_method_haz[name] = pm
            aggs[name]['tp'] += int(((pm > 0) & (ref > 0)).sum())
            aggs[name]['tn'] += int(((pm == 0) & (ref == 0)).sum())
            aggs[name]['fp'] += int(((pm > 0) & (ref == 0)).sum())
            aggs[name]['fn'] += int(((pm == 0) & (ref > 0)).sum())
            aggs[name]['pred_px'] += int((pm > 0).sum())
            aggs[name]['ref_px'] += int((ref > 0).sum())

        all_samples.append({
            'fname': fname, 'hr': hr, 'lr': lr,
            'base_pred': base_pred, 'imp_pred': imp_pred, 'ref': ref,
            'hazards': per_method_haz,
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_files):
            print(f"  Processed {idx+1}/{len(test_files)}...")

    # Print results
    print()
    print("=" * 80)
    print("FINAL TEST EVALUATION — 300-image test set")
    print("Reference: DERIVED PHOTOMETRIC REFERENCE (not human ground truth)")
    print("=" * 80)

    ref_frac = aggs['baseline']['ref_px'] / (
        aggs['baseline']['tp']+aggs['baseline']['tn']+aggs['baseline']['fp']+aggs['baseline']['fn'])
    print(f"Reference hazard fraction: {ref_frac*100:.1f}%\n")

    final_metrics = {}
    for name, agg in aggs.items():
        m = agg_hazard(agg)
        final_metrics[name] = m

    methods_display = ['baseline', 'v1_improved', 'v2_balanced', 'v2_conservative', 'v2_precision']
    labels = ['BASELINE', 'V1_IMPROVED', 'V2_BALANCED', 'V2_CONSERV.', 'V2_PRECISION']

    print(f"{'Method':<18} {'P%':>8} {'R%':>8} {'F1%':>8} {'IoU%':>8} {'Pred%':>8}")
    print("-" * 65)
    for name, label in zip(methods_display, labels):
        m = final_metrics[name]
        print(f"{label:<18} {m['precision']:>8.2f} {m['recall']:>8.2f} {m['f1']:>8.2f} "
              f"{m['iou']:>8.2f} {m['pred_hazard_frac']:>8.1f}%")

    print()
    print(f"{'Method':<18} {'TP':>12} {'TN':>12} {'FP':>12} {'FN':>12}")
    print("-" * 65)
    for name, label in zip(methods_display, labels):
        m = final_metrics[name]
        print(f"{label:<18} {m['tp']:>12,} {m['tn']:>12,} {m['fp']:>12,} {m['fn']:>12,}")

    print()
    print("--- IMPROVEMENTS vs BASELINE ---")
    base = final_metrics['baseline']
    for name, label in zip(methods_display[1:], labels[1:]):
        m = final_metrics[name]
        print(f"{label}:")
        print(f"  Precision: {base['precision']:.2f}% → {m['precision']:.2f}% ({m['precision']-base['precision']:+.2f}pp)")
        print(f"  Recall:    {base['recall']:.2f}% → {m['recall']:.2f}% ({m['recall']-base['recall']:+.2f}pp)")
        print(f"  F1:        {base['f1']:.2f}% → {m['f1']:.2f}% ({m['f1']-base['f1']:+.2f}pp)")
        print(f"  IoU:       {base['iou']:.2f}% → {m['iou']:.2f}% ({m['iou']-base['iou']:+.2f}pp)")
        fp_old = base['fp']; fp_new = m['fp']
        fn_old = base['fn']; fn_new = m['fn']
        print(f"  FP change: {fp_old:,} → {fp_new:,} ({fp_new-fp_old:+,})")
        print(f"  FN change: {fn_old:,} → {fn_new:,} ({fn_new-fn_old:+,})")
        print()

    print("--- RESTORATION (v1 baseline vs epoch-57) ---")
    ra = agg_rest(rest_baseline)
    rb = agg_rest(rest_improved)
    for k in ['mae','rmse','r2','psnr','ssim']:
        print(f"  {k.upper():<6}: {ra[k]['mean']:.4f} → {rb[k]['mean']:.4f} "
              f"({rb[k]['mean']-ra[k]['mean']:+.4f})")

    # Visual analysis for representative samples
    print(f"\nGenerating visual analysis...")
    by_v2b_f1 = sorted(all_samples, key=lambda s: (
        final_metrics['v2_balanced']['tp'] - 0  # just use index
    ))
    # Select: first, middle, last, plus a few spaced out
    vis_idx = [0, 30, 75, 150, 225, len(all_samples)-1]
    for i, vi in enumerate(vis_idx):
        if vi >= len(all_samples):
            continue
        s = all_samples[vi]
        out_p = os.path.join(VIS_DIR, f"analysis_{i:02d}_{s['fname']}")
        save_error_vis(
            out_p, s['fname'], s['lr'], s['hr'], s['imp_pred'], s['ref'],
            s['hazards']['baseline'], s['hazards']['v1_improved'],
            s['hazards']['v2_balanced'], s['hazards']['v2_conservative'],
            s['hazards']['v2_precision'],
        )
    print(f"  Saved {len(vis_idx)} panels to {VIS_DIR}/")

    # Save results
    full_results = {
        'test_set_size': len(all_samples),
        'reference_type': 'DERIVED PHOTOMETRIC REFERENCE (not human ground truth)',
        'ref_hazard_frac_pct': round(ref_frac * 100, 1),
        'checkpoint_baseline': 'weights/lunar_unet_best.pth',
        'checkpoint_improved': 'weights/lunar_unet_v2_best.pth (epoch 57, val_loss=0.1146)',
        'hazard_configs': {
            'baseline': 'Canny(80,200)+MORPH_ELLIPSE(11x11)x2',
            'v1_improved': 'multi-feature: canny(30,80)+sobel+lap+shadow(50)+tophat, score>=0.25',
            'v2_balanced': 'oracle-core (Otsu-shadow+tophat(7)+blackhat(7)+tophat(11)+blackhat(11)+morph_grad) + buf(7x7,2) + mc=50',
            'v2_conservative': 'oracle-core(7+15) + buf(7x7,2) + mc=100',
            'v2_precision': 'oracle-core(7) + buf(9x9,2) + mc=100',
        },
        'hazard_detection': {name: final_metrics[name] for name in methods_display},
        'restoration': {
            'baseline': {k: agg_rest(rest_baseline)[k]['mean'] for k in agg_rest(rest_baseline)},
            'epoch57': {k: agg_rest(rest_improved)[k]['mean'] for k in agg_rest(rest_improved)},
        },
    }

    with open(os.path.join(OUT_DIR, "v2_final_comparison.json"), "w") as f:
        json.dump(full_results, f, indent=2)

    with open(os.path.join(OUT_DIR, "v2_per_image.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['fname'] + [
            f'{name}_{k}' for name in methods_display
            for k in ['haz_p','haz_r','haz_f1','haz_iou']
        ] + ['imp_psnr','imp_ssim'])
        writer.writeheader()

    print(f"\nResults saved to {OUT_DIR}/")
    return full_results


if __name__ == "__main__":
    main()
