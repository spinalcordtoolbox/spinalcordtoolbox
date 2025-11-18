# -*- coding: utf-8 -*-
"""
Inference script used to generate the motion-corrected 4D volumes from DenseRigid model.
"""

import os
import torch
import textwrap
import numpy as np
import nibabel as nib

from skimage.exposure import match_histograms

def run_mocoDL(fname_data, fname_mask, ofolder, mode="fmri", fname_ref=None, fname_bvals=None, fname_bvecs=None):
    from tqdm import tqdm
    from spinalcordtoolbox.mocoDL.model import DenseRigidReg, RigidWarp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(ofolder, exist_ok=True)

    # Resolve checkpoint based on mode
    sct_root = os.environ.get("SCT_DIR", os.path.expanduser("~/.sct"))

    if mode == "fmri":
        ckpt_name = "fmri.ckpt"
    elif mode == "dmri":
        ckpt_name = "dmri.ckpt"

    ckpt_path = os.path.join(sct_root, "spinalcordtoolbox", "mocoDL", "weights", ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[mocoDL] Checkpoint not found: {ckpt_path}")

    print("\n[INFO] Running DL-based motion correction (DenseRigidNet)...\n")

    # Display SCT-style parameter banner
    print(textwrap.dedent(f"""
        Input parameters (DL-based motion correction):
        ----------------------------------------------------
        Input file:            {fname_data}
        Output folder:         {os.path.abspath(ofolder)}
        Mode:                  {mode}
        Mask:                  {fname_mask if fname_mask else 'None'}
        Reference image:       {fname_ref if fname_ref else 'None'}
        bvals (dmri only):     {fname_bvals if fname_bvals else 'None'}
        bvecs (dmri only):     {fname_bvecs if fname_bvecs else 'None'}
    """))

    # Load NIfTI data
    def load_nifti(p):
        img = nib.load(p)
        return img, img.get_fdata().astype(np.float32)

    # Check required inputs
    mov_img, mov_np = load_nifti(fname_data)
    affine, header = mov_img.affine, mov_img.header

    if fname_mask:
        mask_img, mask_np = load_nifti(fname_mask)
    else:
        raise ValueError("[mocoDL] mask_path is required (spinal cord mask).")

    if fname_ref:
        fix_img, fix_np = load_nifti(fname_ref)
    else:
        print("[mocoDL] No reference provided — will use first volume (t=0) of moving data.")
        fix_np = mov_np[..., 0]
        fix_np = np.repeat(fix_np[..., None], mov_np.shape[-1], axis=-1)

    # Load DL model
    print(f"[mocoDL] Loading checkpoint: {ckpt_path}")
    model = DenseRigidReg.load_from_checkpoint(ckpt_path, map_location=device)
    model = model.to(device)
    model.eval()
    model.warp = RigidWarp(mode="nearest")

    moving = torch.from_numpy(mov_np).unsqueeze(0).unsqueeze(0).to(device)
    fixed  = torch.from_numpy(fix_np).unsqueeze(0).unsqueeze(0).to(device)
    mask   = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

    # Run inference
    print("[mocoDL] Starting inference...")
    for _ in tqdm(range(1), desc="Running model"):
        with torch.no_grad():
            warped, Tx, Ty = model(moving, fixed, mask)

    warped = warped.squeeze().cpu().numpy()  # (H,W,D,T)
    Tx = Tx.squeeze().cpu().numpy()  # (D,T)
    Ty = Ty.squeeze().cpu().numpy()  # (D,T)

    H, W, D, T = warped.shape
    sharpened = np.copy(warped)

    # import cv2
    # for t in range(T):
    #     for d in range(D):
    #         img_warped = warped[..., d, t]
    #         img_raw = mov_np[..., d, t]
    #         mask_slice = mask_np[..., d]
    #
    #         if np.count_nonzero(mask_slice) == 0:
    #             sharpened[..., d, t] = img_warped
    #             continue
    #
    #         raw_smooth = cv2.GaussianBlur(img_raw, (0, 0), 0.6)
    #         texture = img_raw - raw_smooth
    #         out = img_warped + 1.3 * texture
    #         lo, hi = np.percentile(img_raw[mask_slice > 0], [0.5, 99.5])
    #         out = np.clip(out, lo, hi)
    #         sharpened[..., d, t] = out
    #
    # sharpened[mask_np == 0] = mov_np[mask_np == 0]

    voxel = header.get_zooms()[:3]
    disp = np.zeros((H, W, D, T, 3), dtype=np.float32)
    for t in range(T):
        for d in range(D):
            disp[..., d, t, 0] = Tx[d, t] * voxel[0]
            disp[..., d, t, 1] = Ty[d, t] * voxel[1]
            disp[..., d, t, 2] = 0.0

    matched = np.zeros_like(sharpened)
    for t in range(T):
        matched[..., t] = match_histograms(sharpened[..., t], mov_np[..., t])

    base = os.path.basename(fname_data).replace(".nii.gz", "").replace(".nii", "")
    out_path = os.path.join(ofolder, f"{base}_mocoDL.nii.gz")
    nib.save(nib.Nifti1Image(matched, affine, header=header), out_path)

    # Save Tx, Ty, and displacement fields
    nib.save(nib.Nifti1Image(Tx[np.newaxis, np.newaxis, ...], affine, header=header),
        os.path.join(ofolder, f"{base}_Tx.nii.gz"))
    nib.save(nib.Nifti1Image(Ty[np.newaxis, np.newaxis, ...], affine, header=header),
        os.path.join(ofolder, f"{base}_Ty.nii.gz"))

    # Save 5D dispfield
    disp5D_img = nib.Nifti1Image(disp, affine, header=header)
    disp5D_img.header.set_intent('vector', (), '')
    nib.save(disp5D_img, os.path.join(ofolder, f"{base}_dispfield-5D.nii.gz"))

    # Save per-timepoint dispfields
    disp_dir = os.path.join(ofolder, "dispfield")
    os.makedirs(disp_dir, exist_ok=True)
    for t in tqdm(range(T), desc="Saving displacement fields"):
        disp_t = disp[..., t, :]  # (H, W, D, 3)
        disp_img = nib.Nifti1Image(disp_t, affine, header=header)
        disp_img.header.set_intent('vector', (), '')
        nib.save(disp_img, os.path.join(disp_dir, f"{base}_dispfield_t{t:04d}.nii.gz"))

    print(f"[mocoDL] Output saved to {out_path}")
    return out_path