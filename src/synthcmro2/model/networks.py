"""Generator (3D attention residual U-Net) and discriminator of the conditional GAN.

Layer names, shapes and the forward pass are identical to ``p_syntheticCMRO2_functions.py``, so the
saved ``state_dict`` files load without conversion (tested in tests/test_networks.py).
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """BN -> LeakyReLU -> 3x3x3 conv, twice, plus the identity (1x1x1 projection if channels change)."""

    def __init__(self, channels_in: int, channels_out: int):
        super().__init__()
        self.use_proj = channels_in != channels_out
        if self.use_proj:
            self.proj = nn.Conv3d(channels_in, channels_out, kernel_size=1, padding=0)
        self.batch1 = nn.BatchNorm3d(channels_in, affine=True)
        self.act = nn.LeakyReLU()  # negative slope 0.01
        self.conv1 = nn.Conv3d(channels_in, channels_out, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm3d(channels_out, affine=True)
        self.conv2 = nn.Conv3d(channels_out, channels_out, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x) if self.use_proj else x
        out = self.conv1(self.act(self.batch1(x)))
        out = self.conv2(self.act(self.batch2(out)))
        return out + identity


class AttentionBlock(nn.Module):
    """Additive attention gate: the decoder signal g weights the encoder features x."""

    def __init__(self, x_channels: int, g_channels: int, inter_channels: int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv3d(g_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(inter_channels),
        )
        self.w_x = nn.Sequential(
            nn.Conv3d(x_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor, return_psi: bool = False):
        psi = self.psi(self.relu(self.w_g(g) + self.w_x(x)))
        out = psi * x
        return (out, psi) if return_psi else out


class UNet(nn.Module):
    """Generator: three encoder levels (64, 128, 256), 512-channel bottleneck, attention-gated skips."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.enc1_res, self.enc1_drop, self.enc1_pool = (
            ResidualBlock(in_channels, 64),
            nn.Dropout3d(0.3),
            nn.MaxPool3d(2),
        )
        self.enc2_res, self.enc2_drop, self.enc2_pool = ResidualBlock(64, 128), nn.Dropout3d(0.3), nn.MaxPool3d(2)
        self.enc3_res, self.enc3_drop, self.enc3_pool = ResidualBlock(128, 256), nn.Dropout3d(0.4), nn.MaxPool3d(2)
        self.bottleneck_res, self.bottleneck_drop = ResidualBlock(256, 512), nn.Dropout3d(0.5)

        self.up7 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.att7 = AttentionBlock(256, 256, 128)
        self.dec7_res = ResidualBlock(512, 256)
        self.up8 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.att8 = AttentionBlock(128, 128, 64)
        self.dec8_res = ResidualBlock(256, 128)
        self.up9 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.att9 = AttentionBlock(64, 64, 32)
        self.dec9_res = ResidualBlock(128, 64)
        self.final_conv = nn.Conv3d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, return_att: bool = False):
        c1 = self.enc1_drop(self.enc1_res(x))
        c2 = self.enc2_drop(self.enc2_res(self.enc1_pool(c1)))
        c3 = self.enc3_drop(self.enc3_res(self.enc2_pool(c2)))
        c4 = self.bottleneck_drop(self.bottleneck_res(self.enc3_pool(c3)))

        up7 = self.up7(c4)
        skip7, psi7 = self.att7(c3, up7, return_psi=True)
        u7 = self.dec7_res(torch.cat([skip7, up7], dim=1))
        up8 = self.up8(u7)
        skip8, psi8 = self.att8(c2, up8, return_psi=True)
        u8 = self.dec8_res(torch.cat([skip8, up8], dim=1))
        up9 = self.up9(u8)
        skip9, psi9 = self.att9(c1, up9, return_psi=True)
        u9 = self.dec9_res(torch.cat([skip9, up9], dim=1))

        out = self.final_conv(u9)
        if return_att:
            return out, {"psi7": psi7, "psi8": psi8, "psi9": psi9}
        return out


class Discriminator(nn.Module):
    """Conditional discriminator on (inputs, CMRO2): three conv levels, global average pooling, two FC layers."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1, self.act1, self.pool1 = (
            nn.Conv3d(in_channels + 1, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(2),
        )
        self.conv2, self.act2, self.pool2 = nn.Conv3d(64, 128, 3, padding=1), nn.LeakyReLU(0.2), nn.MaxPool3d(2)
        self.conv3, self.act3, self.pool3 = nn.Conv3d(128, 256, 3, padding=1), nn.LeakyReLU(0.2), nn.MaxPool3d(2)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1, self.act4, self.fc2 = nn.Linear(256, 1024), nn.LeakyReLU(0.2), nn.Linear(1024, 1)

    def forward(self, inputs: torch.Tensor, cmro2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([inputs, cmro2], dim=1)
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))
        x = self.global_pool(x).view(x.size(0), -1)
        return self.fc2(self.act4(self.fc1(x)))  # logits


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
