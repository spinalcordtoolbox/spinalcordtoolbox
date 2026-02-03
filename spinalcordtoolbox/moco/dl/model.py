#!/usr/bin/env python
#
# DL-based motion correction for dMRI/fMRI
#
# Copyright (c) 2025 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import torch
import torch.nn as nn
from monai.networks.blocks import Warp


# -----------------------------
# Some helper function
# -----------------------------
def _get_fixed_t(fixed: "torch.Tensor", t: int) -> "torch.Tensor":
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
    Slice-wise rigid warp using MONAI's Warp (voxel-unit flow).
    """
    def __init__(self, mode="bilinear", padding_mode="border"):
        super().__init__()
        self.warper = Warp(mode=mode, padding_mode=padding_mode)

    def build_field(self, Tx, Ty, vol):
        """
        Args:
            Tx, Ty: (B,1,D) voxel shifts
            vol: (B,1,H,W,D)
        Returns:
            flow: (B,3,H,W,D) voxel-unit displacement field
        """
        B, C, H, W, D = vol.shape

        # Expand slice-wise translations Tx, Ty into full 3D displacement field
        # Tx_field: (B,1,H,W,D)
        Tx_field = Tx[:, :, None, None, :].expand(B, 1, H, W, D)
        Ty_field = Ty[:, :, None, None, :].expand(B, 1, H, W, D)
        Tz_field = torch.zeros_like(Tx_field)

        return torch.cat([Tx_field, Ty_field, Tz_field], dim=1)

    def forward(self, vol, Tx, Ty):
        """
        vol: (B,1,H,W,D)
        Tx,Ty: (B,1,D)
        """
        flow = self.build_field(Tx, Ty, vol)
        warped = self.warper(vol, flow)     # MONAI Warp does the 3D spatial transform
        return warped, flow


# -----------------------------
# Main Model
# -----------------------------
class DenseRigidReg(nn.Module):
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
        warped_list, flow_list, Tx_list, Ty_list = [], [], [], []

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

            warped_t, flow_t = self.warp(mov_t, Tx, Ty)
            warped_list.append(warped_t)
            flow_list.append(flow_t)
            Tx_list.append(Tx)
            Ty_list.append(Ty)

        warped_all = torch.stack(warped_list, dim=-1)
        flow_all = torch.stack(flow_list, dim=-1)  # (B,H,W,D,3,T)
        Tx_all = torch.stack(Tx_list, dim=-1)  # (B,1,D,T)
        Ty_all = torch.stack(Ty_list, dim=-1)
        return warped_all, flow_all, Tx_all, Ty_all
