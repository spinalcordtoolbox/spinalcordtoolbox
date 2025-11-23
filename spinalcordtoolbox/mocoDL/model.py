# -*- coding: utf-8 -*-
"""
Created on Mon Sep 1 2025
@author: Nontharat Tucksinapinunchai

DenseRigidNet: DenseNet-based Rigid Motion Correction for dMRI/fMRI
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


# -----------------------------
# Some helper function
# -----------------------------
def _get_fixed_t(fixed: torch.Tensor, t: int) -> torch.Tensor:
    """
    Return fixed volume (B,1,H,W,D) at time t, supporting both 3D fixed (fMRI) and 4D fixed (dMRI).
    """
    if fixed.ndim == 5:
        return fixed.contiguous()
    if fixed.ndim == 6:
        if isinstance(t, torch.Tensor):
            t = int(t.item())
        return fixed[..., t].contiguous()
    raise ValueError(f"Unexpected fixed ndim={fixed.ndim}")


def normalize_volume(x, mask, pmin=1, pmax=99, eps=1e-6):
    """
    Normalizes intensity inside mask to 0–1 range using percentile scaling.
    """
    m = (mask > 0).float()
    vals = x[m.bool()]
    if vals.numel() < 10:
        return x  # skip if mask empty
    lo = torch.quantile(vals, pmin/100)
    hi = torch.quantile(vals, pmax/100)
    return ((x - lo) / (hi - lo + eps)).clamp(0, 1) * m


# -----------------------------
# Dense Block and Layer
# -----------------------------
class DenseBlock(nn.Module):
    """
    3D DenseNet block (multiple Dense Layers) with growth connections.
    """
    def __init__(self, in_channels, growth_rate, n_layers=5):
        super().__init__()
        layers, channels = [], in_channels
        for _ in range(n_layers):
            layers.append(nn.Sequential(        # Dense Layer: IN → ReLU → Conv(3x3)
                nn.InstanceNorm3d(channels, affine=True, track_running_stats=False),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels, growth_rate, 3, padding=1, bias=False)
            ))
            channels += growth_rate
        self.block = nn.ModuleList(layers)

    def forward(self, x):
        for lenght in self.block:
            out = lenght(x)
            x = torch.cat([x, out], dim=1)
        return x


# -----------------------------
# DenseNet
# -----------------------------
class DenseNetRegressorSliceWise(nn.Module):
    """
    DenseNet backbone that predicts slice-wise in-plane translation (Tx, Ty) for each z-slice in the 3D volume.
    Output shape: (B, 3, D)
    """
    def __init__(self, in_channels=2, growth_rate=8, num_blocks=2, max_vox_shift=5.0):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, 16, (3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))
        self.blocks = nn.ModuleList()
        self.max_vox_shift = max_vox_shift
        self.shift_scale = nn.Parameter(torch.tensor(1.0))  # learnable >0 via softplus

        ch = 16
        for _ in range(num_blocks):
            blk = DenseBlock(ch, growth_rate)   # DenseBlock
            self.blocks.append(blk)
            ch += growth_rate * 5
            self.blocks.append(nn.Sequential(       # Transition Layer: IN → ReLU → Conv(1x1) → MaxPool(2x2)
                nn.InstanceNorm3d(ch, affine=True, track_running_stats=False),
                nn.ReLU(inplace=True),
                nn.Conv3d(ch, ch // 2, 1, bias=False),
                nn.MaxPool3d((2, 2, 1))     # sharper feature propagation
            ))
            ch = ch // 2

        # keep slice (depth) dimension, pool only H,W
        self.slice_pool = nn.AdaptiveAvgPool3d((1, 1, None))
        self.conv_out = nn.Conv1d(ch, 2, kernel_size=1)  # output Tx, Ty, Th per slice

        # initialize output near zero (identity)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x):
        x = self.conv_in(x)
        for b in self.blocks:
            x = b(x)
        x = x.mean(dim=(2, 3))           # (B, C, D')
        theta = self.conv_out(x)        # (B, 2, D')

        s = torch.nn.functional.softplus(self.shift_scale) + 1e-4
        Ty = torch.tanh(theta[:, 0:1, :]) * (self.max_vox_shift * s)
        Tx = torch.tanh(theta[:, 1:2, :]) * (self.max_vox_shift * s)
        return torch.cat([Tx, Ty], dim=1)


# -----------------------------
# Warp Function
# -----------------------------
class RigidWarp(nn.Module):
    """
    Apply rigid 2D transformation (Tx, Ty) slice-wise. Each slice is translated independently in-plane.
    """
    def __init__(self, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode

    def forward(self, vol, Tx, Ty):
        assert vol.ndim == 5, "vol must be (B,C,H,W,D)"
        B, C, H, W, D = vol.shape
        device = vol.device
        dtype = vol.dtype

        # (B,1,D) -> (B,D)
        Tx = Tx.view(B, D)
        Ty = Ty.view(B, D)

        # Build base 2D grid (sampling grid) in [-1,1], shape (H,W,2)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device, dtype=dtype),
            torch.linspace(-1, 1, W, device=device, dtype=dtype),
            indexing="ij"
        )
        base = torch.stack([xx, yy], dim=-1)  # (H,W,2)

        # Expand to (B,D,H,W,2)
        base = base.view(1, 1, H, W, 2).expand(B, D, H, W, 2)

        # Normalize translations to grid units [-1,1]
        txn = (2.0 * Tx / max(W - 1, 1)).view(B, D, 1, 1, 1)  # (B,D,1,1,1)
        tyn = (2.0 * Ty / max(H - 1, 1)).view(B, D, 1, 1, 1)

        grid = base.clone()
        grid[..., 0] += txn[..., 0]  # x += Tx
        grid[..., 1] += tyn[..., 0]  # y += Ty

        # Reshape for 2D grid_sample
        grid_2d = grid.view(B * D, H, W, 2)  # (B*D,H,W,2)
        vol_2d = vol.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)  # (B*D,C,H,W)

        # Warping
        warped_2d = F.grid_sample(
            vol_2d, grid_2d,
            mode=self.mode, padding_mode="border", align_corners=True
        )
        warped = warped_2d.view(B, D, C, H, W).permute(0, 2, 3, 4, 1).contiguous()  # (B,C,H,W,D)
        return warped


# -----------------------------
# Main Model
# -----------------------------
class DenseRigidReg(pl.LightningModule):
    """
    Main PyTorch Lightning module for DenseNet-based rigid motion correction.
    """
    def __init__(self, lr=1e-4):
        super().__init__()
        self.lr = lr
        self.backbone = DenseNetRegressorSliceWise(in_channels=2, max_vox_shift=3.0)
        self.warp = RigidWarp(mode="bilinear")

    def forward(self, moving, fixed, mask):
        """
            moving: (B,1,H,W,D,T)
            fixed:  (B,1,H,W,D) or (B,1,H,W,D,T)
            mask:   (B,1,H,W,D)
            Returns:
                warped_all: (B,1,H,W,D,T)
                Tx_all, Ty_all: (B,1,D,T)
            """
        B, _, H, W, D, T = moving.shape
        warped_list, Tx_list, Ty_list = [], [], []

        for t in range(T):
            mov_t = moving[..., t]
            fix_t = _get_fixed_t(fixed, t)

            # Normalize at the same resolution as the network input
            mov_norm_ds = normalize_volume(mov_t, mask)
            fix_norm_ds = normalize_volume(fix_t, mask)

            x = torch.cat([mov_norm_ds, fix_norm_ds], dim=1)

            theta = self.backbone(x)  # (B, 2, D')
            Ty = theta[:, 0:1, :]
            Tx = theta[:, 1:2, :]

            warped = self.warp(mov_t, Tx, Ty)
            warped_list.append(warped)
            Tx_list.append(Tx)
            Ty_list.append(Ty)

            # cleanup
            del mov_t, fix_t, mov_norm_ds, fix_norm_ds, x, theta
            torch.cuda.empty_cache()

        warped_all = torch.stack(warped_list, dim=-1)
        Tx_all = torch.stack(Tx_list, dim=-1)  # (B,1,D,T)
        Ty_all = torch.stack(Ty_list, dim=-1)
        return warped_all, Tx_all, Ty_all
