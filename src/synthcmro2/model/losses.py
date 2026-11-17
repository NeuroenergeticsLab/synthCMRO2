"""Losses of the conditional GAN (as in the notebooks)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch import nn


def ssim_value(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """3D SSIM with the data range of the analytical map (whole batch), guarded against a zero range."""
    data_range = max(float(y.max() - y.min()), 1e-12)
    return ssim(y_hat, y, data_range=data_range, size_average=True)


def generator_loss(
    y: torch.Tensor, y_hat: torch.Tensor, adv_logits: torch.Tensor, weights: dict[str, float], bce: nn.Module
) -> torch.Tensor:
    """L_G = w_l1 * L1 + w_adv * BCE(D(x, y_hat), 1) + w_ssim * (1 - SSIM); weights 1, 0.01, 0.5."""
    l1 = F.l1_loss(y_hat, y)
    adv = bce(adv_logits, torch.ones_like(adv_logits))
    return weights["l1"] * l1 + weights["adversarial"] * adv + weights["ssim"] * (1 - ssim_value(y_hat, y))


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor, bce: nn.Module) -> torch.Tensor:
    """Mean of BCE(real, 1) and BCE(fake, 0)."""
    return 0.5 * (bce(real_logits, torch.ones_like(real_logits)) + bce(fake_logits, torch.zeros_like(fake_logits)))
