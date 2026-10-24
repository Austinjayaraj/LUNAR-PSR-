"""
Unit tests for evaluation metrics module using standard unittest.
Verifies scientific metric implementations on synthetic data,
edge cases, perfect predictions, and zero-division safety.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.metrics import calculate_restoration_metrics, calculate_hazard_metrics


class TestRestorationMetrics(unittest.TestCase):
    def test_perfect_prediction(self):
        """Perfect prediction must produce MAE=0, RMSE=0, R²=1, SSIM=1, PSNR=inf."""
        gt = np.random.randint(20, 240, size=(64, 64), dtype=np.uint8)
        pred = gt.copy()

        metrics = calculate_restoration_metrics(pred, gt)

        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["r2"], 1.0)
        self.assertEqual(metrics["ssim"], 1.0)
        self.assertTrue(math.isinf(metrics["psnr"]))

    def test_constant_offset(self):
        """Constant offset of 10 should produce MAE=10, RMSE=10."""
        gt = np.full((32, 32), 100, dtype=np.uint8)
        pred = np.full((32, 32), 110, dtype=np.uint8)

        metrics = calculate_restoration_metrics(pred, gt)

        self.assertEqual(metrics["mae"], 10.0)
        self.assertEqual(metrics["rmse"], 10.0)
        self.assertTrue(0 < metrics["psnr"] < float("inf"))

    def test_r2_and_ssim_behavior(self):
        """Degraded / noisy prediction must yield lower R² and SSIM."""
        np.random.seed(42)
        gt = np.random.normal(128, 30, size=(128, 128)).clip(0, 255).astype(np.uint8)

        # Mildly noisy version
        noisy_pred = (gt.astype(np.float32) + np.random.normal(0, 5, size=gt.shape)).clip(0, 255).astype(np.uint8)
        metrics_mild = calculate_restoration_metrics(noisy_pred, gt)

        # Severely corrupted version
        very_noisy = np.random.randint(0, 256, size=gt.shape, dtype=np.uint8)
        metrics_severe = calculate_restoration_metrics(very_noisy, gt)

        self.assertGreater(metrics_mild["ssim"], 0.8)
        self.assertLess(metrics_mild["ssim"], 1.0)
        self.assertGreater(metrics_mild["r2"], 0.8)
        self.assertLess(metrics_mild["mae"], metrics_severe["mae"])
        self.assertLess(metrics_mild["rmse"], metrics_severe["rmse"])
        self.assertGreater(metrics_mild["psnr"], metrics_severe["psnr"])
        self.assertGreater(metrics_mild["ssim"], metrics_severe["ssim"])

    def test_shape_mismatch_raises(self):
        """Shape mismatch between prediction and ground truth must raise ValueError."""
        gt = np.zeros((64, 64), dtype=np.uint8)
        pred = np.zeros((32, 32), dtype=np.uint8)

        with self.assertRaises(ValueError):
            calculate_restoration_metrics(pred, gt)


class TestHazardMetrics(unittest.TestCase):
    def test_perfect_hazard_prediction(self):
        """Identical masks should give Accuracy=1, Precision=1, Recall=1, F1=1, IoU=1."""
        gt_mask = np.zeros((10, 10), dtype=np.uint8)
        gt_mask[2:6, 2:6] = 1
        pred_mask = gt_mask.copy()

        metrics = calculate_hazard_metrics(pred_mask, gt_mask)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["tp"], 16)
        self.assertEqual(metrics["tn"], 84)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)

    def test_confusion_matrix_calculation(self):
        """Verify explicit TP, TN, FP, FN calculations."""
        pred_mask = np.array([1, 0, 1, 0], dtype=np.uint8)
        gt_mask = np.array([1, 0, 0, 1], dtype=np.uint8)

        metrics = calculate_hazard_metrics(pred_mask, gt_mask)

        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["iou"], round(1.0 / 3.0, 4))

    def test_zero_division_all_negative(self):
        """When both prediction and ground truth have no positives, division by zero is handled safely."""
        gt_mask = np.zeros((20, 20), dtype=np.uint8)
        pred_mask = np.zeros((20, 20), dtype=np.uint8)

        metrics = calculate_hazard_metrics(pred_mask, gt_mask)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)
        self.assertEqual(metrics["iou"], 0.0)

    def test_zero_division_no_overlap(self):
        """When predicted positives do not overlap ground truth positives."""
        gt_mask = np.zeros((10, 10), dtype=np.uint8)
        gt_mask[0:2, 0:2] = 1  # 4 positives

        pred_mask = np.zeros((10, 10), dtype=np.uint8)
        pred_mask[5:7, 5:7] = 1  # 4 different positives

        metrics = calculate_hazard_metrics(pred_mask, gt_mask)

        self.assertEqual(metrics["tp"], 0)
        self.assertEqual(metrics["fp"], 4)
        self.assertEqual(metrics["fn"], 4)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)
        self.assertEqual(metrics["iou"], 0.0)

    def test_continuous_output_rejected(self):
        """Passing continuous values directly to hazard metrics must raise ValueError."""
        pred_continuous = np.array([[0.2, 0.8], [0.5, 0.9]], dtype=np.float32)
        gt_mask = np.array([[0, 1], [0, 1]], dtype=np.uint8)

        with self.assertRaises(ValueError):
            calculate_hazard_metrics(pred_continuous, gt_mask)


if __name__ == "__main__":
    unittest.main()
