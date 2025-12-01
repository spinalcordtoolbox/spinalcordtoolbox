#!/usr/bin/env python
#
# Inference of DL-based motion correction for dMRI/fMRI
#
# Copyright (c) 2025 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import os
import time
import textwrap
import numpy as np

from spinalcordtoolbox.math import smooth
from spinalcordtoolbox.image import add_suffix, generate_output_file, Image
from spinalcordtoolbox.scripts import sct_dmri_separate_b0_and_dwi
from spinalcordtoolbox.utils.sys import sct_dir_local_path, LazyLoader
from spinalcordtoolbox.moco.dl.model import DenseRigidReg, RigidWarp

torch = LazyLoader("torch", globals(), "torch")
ski_exposure = LazyLoader("ski_exposure", globals(), "skimage.exposure")


def check_dl_args(argv):
    # mask (-m) is mandatory for DL module
    if "-m" not in argv:
        raise ValueError("[moco-dl] Missing required argument: -m <mask>. A spinal cord mask is required for DL-based motion correction.")

    if "-ref" not in argv:
        print("[WARNING] No -ref provided. DL module will use the first volume (t=0) of input as reference.")

    # check if the raw user arguments contain any forbidden args
    forbidden = []
    for forbidden_arg in ["-g", "-x", "-param", "-bvalmin"]:
        if any(arg == forbidden_arg for arg in argv):
            forbidden.append(forbidden_arg)

    if forbidden:
        raise ValueError(
            "The following options cannot be used together with -dl (DL-based motion correction): "
            + ", ".join(forbidden)
            + "\nDL module does not support b-value threshold (-bvalmin), grouping (-g), "
              "final interpolation (-x), or advanced ANTs parameters (-param)."
        )


def moco_dl(fname_data, fname_mask, ofolder, mode="fmri", fname_ref=None, fname_bvals=None, fname_bvecs=None):
    """
        Deep-learning motion correction (DenseRigidNet) for dMRI/fMRI.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(ofolder, exist_ok=True)

    ckpt_path = sct_dir_local_path("data", "moco-dl_models", f"{mode}.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[moco-dl] Checkpoint not found: {ckpt_path}")

    print("\n[INFO] Running DL-based motion correction (DenseRigidNet)...\n")
    # Display parameter summary
    print(textwrap.dedent(f"""
        Input parameters (DL-based motion correction):
        ----------------------------------------------------
        Input file:            {fname_data}
        Output folder:         {os.path.abspath(ofolder)}
        Mode:                  {mode}
        Mask:                  {fname_mask if fname_mask else 'None'}
        Reference image:       {fname_ref if fname_ref else 'first volume of input (t=0)'}
        bvecs (dmri only):     {fname_bvecs if fname_bvecs else 'None'}
        bvals (dmri only):     {fname_bvals if fname_bvals else 'None'}
    """))

    # Load image
    mov_img = Image(fname_data)
    mov_img.change_orientation('RPI')
    mov_np = mov_img.data.astype(np.float32)
    affine, header = mov_img.affine, mov_img.header

    mask_img = Image(fname_mask)
    mask_img.change_orientation('RPI')
    mask_np = mask_img.data.astype(np.float32)

    if fname_ref:
        ref_img = Image(fname_ref)
        ref_img.change_orientation('RPI')
        ref_np = ref_img.data.astype(np.float32)
    else:
        ref_np = mov_np[..., 0]

    # Load DL model
    print(f"[moco-dl] Loading checkpoint: {ckpt_path}")
    model = DenseRigidReg()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model = model.to(device)
    model.eval()
    model.warp = RigidWarp(mode="bilinear")

    moving = torch.from_numpy(mov_np).unsqueeze(0).unsqueeze(0).to(device)
    fixed = torch.from_numpy(ref_np).unsqueeze(0).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)

    # Run inference
    start_time = time.time()
    print("[moco-dl] Starting inference...")
    with torch.no_grad():
        warped, flow, Tx, Ty = model(moving, fixed, mask)

    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.2f} sec")

    warped = warped.squeeze().cpu().numpy()  # (H,W,D,T)
    flow = flow.squeeze().cpu().numpy()
    Tx = Tx.squeeze().cpu().numpy()  # (D,T)
    Ty = Ty.squeeze().cpu().numpy()  # (D,T)

    H, W, D, T = warped.shape
    # Restore high-frequency detail from raw data
    sharpened = np.copy(warped)
    for t in range(T):
        for d in range(D):
            img_warped = warped[..., d, t]
            img_raw = mov_np[..., d, t]
            mask_slice = mask_np[..., d]

            if np.count_nonzero(mask_slice) == 0:
                sharpened[..., d, t] = img_warped
                continue

            raw_smooth = smooth(img_raw.astype(np.float32), sigmas=[0.5, 0.5])
            texture = img_raw - raw_smooth
            out = img_warped + 1.2 * texture
            lo, hi = np.percentile(img_raw[mask_slice > 0], [0.5, 99.5])
            out = np.clip(out, lo, hi)
            sharpened[..., d, t] = out

    matched = np.zeros_like(sharpened)
    for t in range(T):
        matched[..., t] = ski_exposure.match_histograms(
            sharpened[..., t].astype(np.float32),
            mov_np[..., t].astype(np.float32)
        )

    # Save Moco output
    tmp_main = os.path.join(ofolder, "tmp_mocoDL.nii.gz")
    im_main = Image(matched, hdr=header).save(tmp_main)
    im_main.affine = affine
    im_main.save(tmp_main)

    base_output = add_suffix(fname_data, "_mocoDL")
    fname_moco = generate_output_file(tmp_main, os.path.join(ofolder, base_output))

    # Save mean output
    print("[moco-dl] Computing time-averaged output volume...")
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
        Image(mean_vol, hdr=header).save(tmp_mean)
        generate_output_file(tmp_mean, add_suffix(fname_moco, "_mean"))

    # Save Tx, Ty, and displacement fields
    print("[moco-dl] Saving translation params...")

    Tx_img_path_tmp = os.path.join(ofolder, "tmp_Tx.nii.gz")
    Ty_img_path_tmp = os.path.join(ofolder, "tmp_Ty.nii.gz")

    im_Tx = Image(Tx[np.newaxis, np.newaxis, ...], hdr=header)
    im_Tx.hdr.set_data_shape(im_Tx.data.shape)
    im_Tx.save(Tx_img_path_tmp)
    im_Ty = Image(Ty[np.newaxis, np.newaxis, ...], hdr=header)
    im_Ty.hdr.set_data_shape(im_Ty.data.shape)
    im_Ty.save(Ty_img_path_tmp)

    generate_output_file(Tx_img_path_tmp, add_suffix(fname_moco, "_Tx"))
    generate_output_file(Ty_img_path_tmp, add_suffix(fname_moco, "_Ty"))

    # Save 5D dispfield
    print("[moco-dl] Saving displacement fields...")
    disp5D = np.moveaxis(flow, 0, -1)  # (H,W,D,T,3)
    tmp_disp5D = os.path.join(ofolder, "tmp_dispfield5D.nii.gz")

    im_disp5D = Image(disp5D, hdr=header)
    im_disp5D.hdr.set_data_shape(disp5D.shape)
    im_disp5D.hdr.set_intent('vector', (), '')
    im_disp5D.affine = affine
    im_disp5D.save(tmp_disp5D)

    generate_output_file(tmp_disp5D, add_suffix(fname_moco, "_dispfield-all"))

    # Save per-timepoint displacement fields
    disp_dir = os.path.join(ofolder, "dispfield")
    os.makedirs(disp_dir, exist_ok=True)

    print("[moco-dl] Saving displacement fields per timepoint...")
    for t in range(T):
        disp_t = disp5D[..., t, :]  # (H,W,D,3)
        im_disp_t = Image(disp_t, hdr=header)
        im_disp_t.hdr.set_data_shape(disp_t.shape)
        im_disp_t.hdr.set_intent('vector', (), '')
        im_disp_t.affine = affine
        im_disp_t.save(os.path.join(disp_dir, f"warp_t{t:04d}.nii.gz"))

    print(f"[moco-dl] Outputs saved in: {os.path.abspath(ofolder)}")
    return fname_moco
