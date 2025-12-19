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
from shutil import copyfile

from spinalcordtoolbox.math import smooth
from spinalcordtoolbox.image import add_suffix, generate_output_file, Image
from spinalcordtoolbox.scripts import sct_dmri_separate_b0_and_dwi
from spinalcordtoolbox.utils.sys import sct_dir_local_path, LazyLoader, printv
from spinalcordtoolbox.utils.fs import tmp_create, extract_fname, rmtree

torch = LazyLoader("torch", globals(), "torch")
ski_exposure = LazyLoader("ski_exposure", globals(), "skimage.exposure")
moco_dl_model = LazyLoader("moco_dl_model", globals(), "spinalcordtoolbox.moco.dl.model")


def check_dl_args(argv):
    # mask (-m) is mandatory for DL module
    if "-m" not in argv:
        raise ValueError("\n[moco-dl] Missing required argument: -m <mask>. A spinal cord mask is required for DL-based motion correction.")

    if "-ref" not in argv:
        raise ValueError("\n[moco-dl] Missing required argument: -ref <reference>. A target image (3D or 4D) is required for DL-based motion correction.")

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


def moco_dl(fname_data, fname_mask='', fname_ref='', path_out='', mode="fmri", fname_bvals='', fname_bvecs=''):
    """
        Deep-learning motion correction (DenseRigidNet) for dMRI/fMRI.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(path_out, exist_ok=True)

    ckpt_path = sct_dir_local_path("data", "moco-dl_models", f"{mode}.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"\n[moco-dl] MocoDL models not found. Please install them using sct_download_data -d moco-dl_models.")

    printv("\n[INFO] Running DL-based motion correction (DenseRigidNet)...\n")
    # Display parameter summary
    printv(textwrap.dedent(f"""
        Input parameters (DL-based motion correction):
        ----------------------------------------------------
        Input file:            {fname_data}
        Output folder:         {os.path.abspath(path_out)}
        Mode:                  {mode}
        Mask:                  {fname_mask if fname_mask != '' else 'None'}
        Reference image:       {fname_ref if fname_ref != '' else 'None'}
        bvecs (dmri only):     {fname_bvecs}
        bvals (dmri only):     {fname_bvals}
    """))

    # Create tmp folder
    path_tmp = tmp_create(basename="moco-dl")

    # Copying input data to tmp folder
    printv("\nCopying input data to tmp folder...")
    im_data = Image(fname_data)
    affine, header = im_data.affine, im_data.header
    im_data.save(os.path.join(path_tmp))
    mov_img = Image(im_data).data.astype(np.float32)
    if fname_mask != '':
        im_mask = Image(fname_mask)
        im_mask.save(os.path.join(path_tmp))
        mask_img = Image(im_mask).data.astype(np.float32)
    if fname_ref != '':
        im_ref = Image(fname_ref)
        im_ref.save(os.path.join(path_tmp))
        ref_img = Image(im_ref).data.astype(np.float32)
    if fname_bvals != '':
        _, _, ext_bvals = extract_fname(fname_bvals)
        file_bvals = f"bvals.{ext_bvals}"  # Use hardcoded name to avoid potential duplicate files when copying
        copyfile(fname_bvals, os.path.join(path_tmp, file_bvals))
        fname_bvals = file_bvals
    if fname_bvecs != '':
        _, _, ext_bvecs = extract_fname(fname_bvecs)
        file_bvecs = f"bvecs.{ext_bvecs}"  # Use hardcoded name to avoid potential duplicate files when copying
        copyfile(fname_bvecs, os.path.join(path_tmp, file_bvecs))
        fname_bvecs = file_bvecs

    # Build absolute output path and go to tmp folder
    curdir = os.getcwd()
    path_out_abs = os.path.abspath(path_out)
    os.chdir(path_tmp)

    # Load DL model
    printv(f"\n[moco-dl] Loading checkpoint: {ckpt_path}")
    model = moco_dl_model.DenseRigidReg()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model = model.to(device)
    model.eval()
    model.warp = moco_dl_model.RigidWarp(mode="bilinear")

    moving = torch.from_numpy(mov_img).unsqueeze(0).unsqueeze(0).to(device)
    fixed = torch.from_numpy(ref_img).unsqueeze(0).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask_img).unsqueeze(0).unsqueeze(0).to(device)

    # Run inference
    start_time = time.time()
    printv("\n[moco-dl] Starting inference...")
    with torch.no_grad():
        warped, flow, Tx, Ty = model(moving, fixed, mask)

    inference_time = time.time() - start_time
    printv(f"\nInference time: {inference_time:.2f} sec")

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
            img_raw = mov_img[..., d, t]
            mask_slice = mask_img[..., d]

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
            mov_img[..., t].astype(np.float32)
        )

    # Save Moco output
    im_moco = Image(matched, hdr=header)
    im_moco.affine = affine
    fname_moco_tmp = os.path.join(path_tmp, "mocoDL.nii.gz")
    im_moco.save(fname_moco_tmp)

    # Save mean output
    printv("\n[moco-dl] Computing time-averaged output volume...")
    if mode == "dmri":
        args = ['-i', fname_moco_tmp, '-bvec', fname_bvecs, '-a', '1', '-v', '0']
        if fname_bvals:
            args += ['-bval', fname_bvals]
        _, fname_b0_mean, _, fname_dwi_mean = sct_dmri_separate_b0_and_dwi.main(argv=args)
    else:
        fname_moco_mean = add_suffix(fname_moco_tmp, '_mean')
        im_moco.mean(dim=3).save(fname_moco_mean)

    # Save Tx, Ty, and displacement fields
    printv("\n[moco-dl] Saving translation params...")
    im_Tx = Image(Tx[np.newaxis, np.newaxis, ...], hdr=header)
    im_Tx.hdr.set_data_shape(im_Tx.data.shape)
    tx_tmp = os.path.join(path_tmp, "Tx.nii.gz")
    im_Tx.save(tx_tmp)
    im_Ty = Image(Ty[np.newaxis, np.newaxis, ...], hdr=header)
    im_Ty.hdr.set_data_shape(im_Ty.data.shape)
    ty_tmp = os.path.join(path_tmp, "Ty.nii.gz")
    im_Ty.save(ty_tmp)

    # Save 5D dispfield
    printv("\n[moco-dl] Saving displacement fields...")
    disp5D = np.moveaxis(flow, 0, -1)  # (H,W,D,T,3)
    im_disp5D = Image(disp5D, hdr=header)
    im_disp5D.hdr.set_data_shape(disp5D.shape)
    im_disp5D.hdr.set_intent('vector', (), '')
    im_disp5D.affine = affine
    disp_tmp = os.path.join(path_tmp, "displacement-field.nii.gz")
    im_disp5D.save(disp_tmp)

    # Generate output files
    printv('\nGenerate output files...')
    # motion corrected data
    fname_moco = os.path.join(path_out_abs, add_suffix(os.path.basename(fname_data), "_mocoDL"))
    generate_output_file(fname_moco_tmp, fname_moco)
    # mean volume
    if mode == "dmri":
        generate_output_file(fname_b0_mean, add_suffix(fname_moco, "_b0_mean"))
        generate_output_file(fname_dwi_mean, add_suffix(fname_moco, "_dwi_mean"))
    else:
        generate_output_file(fname_moco_mean, add_suffix(fname_moco, "_mean"))
    # rigid translation parameter (Tx, Ty)
    generate_output_file(tx_tmp, add_suffix(fname_moco, "_Tx"))
    generate_output_file(ty_tmp, add_suffix(fname_moco, "_Ty"))
    # 5D displacement field
    generate_output_file(disp_tmp, add_suffix(fname_moco, "_dispfield"))

    # Delete temporary files
    printv('\nDelete temporary files...')
    rmtree(path_tmp)

    # come back to working directory
    os.chdir(curdir)

    printv(f"\n[moco-dl] Outputs saved in: {os.path.abspath(path_out)}")
    return fname_moco
