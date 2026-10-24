"""
Loss functions for lunar image restoration.

Combined L1 + SSIM + Edge-aware loss.
- L1 (Charbonnier) for pixel accuracy without MSE's gradient saturation
- SSIM for structural similarity preservation
- Edge loss to preserve crater rims, boulders, and high-frequency terrain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Smooth L1-like loss: sqrt(x^2 + eps^2). Better gradient behavior near zero than L1."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss. Returns 1 - SSIM so it can be minimized."""

    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)
        self.register_buffer("window", window.unsqueeze(0).unsqueeze(0))

    def forward(self, pred, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pad = self.window_size // 2

        w = self.window.to(pred.device)

        mu_pred = F.conv2d(pred, w, padding=pad)
        mu_target = F.conv2d(target, w, padding=pad)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_cross = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred ** 2, w, padding=pad) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, w, padding=pad) - mu_target_sq
        sigma_cross = F.conv2d(pred * target, w, padding=pad) - mu_cross

        ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / (
            (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
        )
        return 1.0 - ssim_map.mean()


class EdgeLoss(nn.Module):
    """
    Penalizes differences in image gradient magnitude.
    Uses Sobel filters to extract edges, then computes L1 between predicted and target edges.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer("sobel_y", sobel_y.unsqueeze(0).unsqueeze(0))

    def forward(self, pred, target):
        sx = self.sobel_x.to(pred.device)
        sy = self.sobel_y.to(pred.device)

        pred_gx = F.conv2d(pred, sx, padding=1)
        pred_gy = F.conv2d(pred, sy, padding=1)
        target_gx = F.conv2d(target, sx, padding=1)
        target_gy = F.conv2d(target, sy, padding=1)

        pred_edge = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-8)
        target_edge = torch.sqrt(target_gx ** 2 + target_gy ** 2 + 1e-8)

        return F.l1_loss(pred_edge, target_edge)


class CombinedRestorationLoss(nn.Module):
    """
    Combined loss: Charbonnier + SSIM + Edge.

    Default weights:
      0.5 * Charbonnier (pixel fidelity)
      0.3 * SSIM (structural preservation)
      0.2 * Edge (high-frequency feature preservation)
    """

    def __init__(self, w_pixel=0.5, w_ssim=0.3, w_edge=0.2):
        super().__init__()
        self.pixel_loss = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgeLoss()
        self.w_pixel = w_pixel
        self.w_ssim = w_ssim
        self.w_edge = w_edge

    def forward(self, pred, target):
        l_pixel = self.pixel_loss(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        l_edge = self.edge_loss(pred, target)
        total = self.w_pixel * l_pixel + self.w_ssim * l_ssim + self.w_edge * l_edge
        return total, {"pixel": l_pixel.item(), "ssim": l_ssim.item(), "edge": l_edge.item()}
