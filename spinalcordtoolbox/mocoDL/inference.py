# -*- coding: utf-8 -*-
"""
Inference script used to generate the motion-corrected 4D volumes from DenseRigid model.
"""

import os
import torch
import textwrap
import numpy as np
import nibabel as nib

from tqdm import tqdm
from skimage.exposure import match_histograms
from spinalcordtoolbox.image import add_suffix, generate_output_file
from spinalcordtoolbox.scripts import sct_dmri_separate_b0_and_dwi
from spinalcordtoolbox.mocoDL.model import DenseRigidReg, RigidWarp

def run_mocoDL(fname_data, fname_mask, ofolder, mode="fmri", fname_ref=None, fname_bvals=None, fname_bvecs=None):
    """
        Deep-learning motion correction (DenseRigidNet) for dMRI/fMRI.
    """
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
    # Display parameter summary
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
    model.warp = RigidWarp(mode="bilinear")

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
    import cv2
    for t in range(T):
        for d in range(D):
            img_warped = warped[..., d, t]
            img_raw = mov_np[..., d, t]
            mask_slice = mask_np[..., d]

            if np.count_nonzero(mask_slice) == 0:
                sharpened[..., d, t] = img_warped
                continue

            raw_smooth = cv2.GaussianBlur(img_raw, (0, 0), 0.6)
            texture = img_raw - raw_smooth
            out = img_warped + 1.3 * texture
            lo, hi = np.percentile(img_raw[mask_slice > 0], [0.5, 99.5])
            out = np.clip(out, lo, hi)
            sharpened[..., d, t] = out

    sharpened[mask_np == 0] = mov_np[mask_np == 0]

    sx, sy, sz = header.get_zooms()[:3]
    disp = np.zeros((H, W, D, T, 3), dtype=np.float32)
    for t in range(T):
        for d in range(D):
            disp[..., d, t, 0] = -Tx[d, t] * sx
            disp[..., d, t, 1] = -Ty[d, t] * sy
            disp[..., d, t, 2] = 0.0

    matched = np.zeros_like(sharpened)
    for t in range(T):
        matched[..., t] = match_histograms(sharpened[..., t], mov_np[..., t])

    # Save Moco output
    base_output = add_suffix(os.path.basename(fname_data), "_mocoDL")
    fname_moco_out = os.path.join(ofolder, base_output)
    tmp_main = os.path.join(ofolder, "tmp_mocoDL.nii.gz")
    nib.save(nib.Nifti1Image(matched, affine, header), tmp_main)
    fname_moco = generate_output_file(tmp_main, fname_moco_out)

    # Save mean output
    print("[mocoDL] Computing time-averaged output volume...")
    if mode == "dmri":
        args = ['-i', fname_moco, '-bvec', fname_bvecs, '-a', '1', '-v', '0']
        if fname_bvals:
            args += ['-bval', fname_bvals]

        _, fname_b0_mean, _, fname_dwi_mean = sct_dmri_separate_b0_and_dwi.main(argv=args)

        generate_output_file(fname_b0_mean, add_suffix(fname_moco, "_b0_mean"))
        generate_output_file(fname_dwi_mean, add_suffix(fname_moco, "_dwi_mean"))
    else:
        mean_vol = np.mean(matched, axis=3)
        tmp_mean = os.path.join(ofolder, "tmp_mean.nii.gz")
        nib.save(nib.Nifti1Image(mean_vol, affine, header), tmp_mean)
        generate_output_file(tmp_mean, add_suffix(fname_moco, "_mean"))

    # Save Tx, Ty, and displacement fields
    print("[mocoDL] Saving translation params and displacement fields...")

    Tx_img_path_tmp = os.path.join(ofolder, "tmp_Tx.nii.gz")
    Ty_img_path_tmp = os.path.join(ofolder, "tmp_Ty.nii.gz")

    nib.save(nib.Nifti1Image(Tx[np.newaxis, np.newaxis, ...], affine, header),
             Tx_img_path_tmp)
    nib.save(nib.Nifti1Image(Ty[np.newaxis, np.newaxis, ...], affine, header),
             Ty_img_path_tmp)
    generate_output_file(Tx_img_path_tmp, add_suffix(fname_moco, "_Tx"))
    generate_output_file(Ty_img_path_tmp, add_suffix(fname_moco, "_Ty"))

    # Save 5D dispfield
    tmp_disp5D = os.path.join(ofolder, "tmp_dispfield5D.nii.gz")
    disp5D_img = nib.Nifti1Image(disp, affine, header)
    disp5D_img.header.set_intent('vector', (), '')
    nib.save(disp5D_img, tmp_disp5D)
    generate_output_file(tmp_disp5D, add_suffix(fname_moco, "_dispfield-all"))

    # Save per-timepoint displacement fields
    disp_dir = os.path.join(ofolder, "dispfield")
    os.makedirs(disp_dir, exist_ok=True)
    for t in tqdm(range(T), desc="Saving displacement fields"):
        disp_t = disp[..., t, :]
        disp_img = nib.Nifti1Image(disp_t, affine, header)
        disp_img.header.set_intent('vector', (), '')
        nib.save(disp_img, os.path.join(disp_dir, f"warp_t{t:04d}.nii.gz"))

    print(f"[mocoDL] Outputs saved in: {ofolder}")
    return fname_moco