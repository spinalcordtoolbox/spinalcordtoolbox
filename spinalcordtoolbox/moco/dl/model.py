#!/usr/bin/env python
#
# DL-based motion correction for dMRI/fMRI
#
# Copyright (c) 2025 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Warp Function
# -----------------------------
def params_to_affine_matrix(params: torch.Tensor) -> torch.Tensor:
    """
    Build a (B, 2, 3) translation affine matrix from (tx, ty) params.
    Fixes rotation to 0 deg (identity matrix [[1, 0], [0, 1]]) and applies only spatial shift.
    """
    tx, ty = params[:, 0], params[:, 1]

    ones = torch.ones_like(tx)
    zeros = torch.zeros_like(tx)

    row0 = torch.stack([ones, zeros, tx], dim=1)
    row1 = torch.stack([zeros, ones, ty], dim=1)
    return torch.stack([row0, row1], dim=1)  # (B, 2, 3)


def warp_with_affine(image: torch.Tensor, affine_matrix: torch.Tensor,
                     mode: str = "bilinear") -> torch.Tensor:
    """
    Apply a (B, 2, 3) affine matrix to a (B, C, H, W) image via affine_grid + grid_sample.
    """
    grid = F.affine_grid(affine_matrix, image.shape, align_corners=False)
    return F.grid_sample(image, grid, mode=mode, padding_mode="zeros", align_corners=False)


# -----------------------------
# Main Model
# -----------------------------
class RigidRegistrationNet(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 8, 2)  # tx, ty only

        # Initialize near identity (zero translation)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def predict_params(self, moving: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        """Returns raw (B, 2) params: tx, ty (normalized [-1, 1] coords)."""
        x = torch.cat([moving, fixed], dim=1)
        feat = self.encoder(x)
        feat = self.pool(feat).flatten(1)
        return self.fc(feat)

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor):
        """
        Returns (warped_moving, affine_matrix) where affine_matrix is (B, 2, 3),
        matching the convention expected by F.affine_grid.
        """
        params = self.predict_params(moving, fixed)
        affine_matrix = params_to_affine_matrix(params)
        warped = warp_with_affine(moving, affine_matrix)
        return warped, affine_matrix
        