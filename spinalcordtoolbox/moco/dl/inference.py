#!/usr/bin/env python
#
# Inference of DL-based motion correction for dMRI/fMRI
#
# Copyright (c) 2025 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

# Two-stage, slice-wise, translation-only motion correction, ported from
# https://github.com/ivadomed/moco-dl (fmri/infer.py, dmri/infer_adaptive.py):
#   1. Reorient to RPI and crop around the spinal cord with sc_crop on the first volume (default margins),
#      as for the training data. Slices outside the box are left uncorrected.
#   2. Stage 1: model 1 against the reference (temporal mean, or -ref), with a non-regression check.
#   3. Stage 2: model 2, iterated with early stopping and the same non-regression check.
#   4. The composed translation is applied once to each slice of the uncropped image.

import os
import time
import textwrap
import numpy as np
import nibabel as nib

from spinalcordtoolbox.image import add_suffix, generate_output_file, Image
from spinalcordtoolbox.scripts import sct_dmri_separate_b0_and_dwi
from spinalcordtoolbox.utils.sys import sct_dir_local_path, LazyLoader, printv
from spinalcordtoolbox.utils.fs import tmp_create, rmtree

torch = LazyLoader("torch", globals(), "torch")
moco_dl_model = LazyLoader("moco_dl_model", globals(), "spinalcordtoolbox.moco.dl.model")
sc_crop = LazyLoader("sc_crop", globals(), "spinalcordtoolbox.sc_crop")


def check_dl_args(argv):
    # check if the raw user arguments contain any forbidden args
    forbidden = []
    for forbidden_arg in ["-m", "-g", "-param", "-x", "-bvalmin"]:
        if any(arg == forbidden_arg for arg in argv):
            forbidden.append(forbidden_arg)

    if forbidden:
        raise ValueError(
            f"The following options cannot be used together with -dl (DL-based motion correction): {', '.join(forbidden)}\n"
            "DL module does not support masks (-m, the spinal cord is located automatically with sc_crop), "
            "b-value threshold (-bvalmin), grouping (-g), final interpolation (-x), "
            "or advanced ANTs parameters (-param)."
        )


# --------------------------------------------------------------------------
# Config -- MUST match the architecture/preprocessing used to train the models
# (ported from moco-dl infer_fmri.py)
# --------------------------------------------------------------------------
TARGET_SHAPE = (64, 64)
BATCH = 4096    # pairs per forward pass / warp call (memory bound only; results do not depend on it)
IDENTITY_THETA = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
N_ITER2 = 6          # maximum number of stage-2 iterations
TOL2 = 1e-3          # stage-2 convergence threshold on mean |residual - identity|
NON_REG_TOL = 0.0    # tolerated NCC decrease before a prediction is rejected
INTERP = "bilinear"  # interpolation of the final warp
MIN_VOLUMES = 2

# sc_crop margins (mm) around the detected cord: the sc_crop defaults used to crop the training data,
# pinned here so that a change of sc_crop's defaults does not change what the models see.
SC_CROP_PAD = dict(pad_superior=40.0, pad_inferior=100.0, pad_left=15.0, pad_right=15.0,
                   pad_anterior=15.0, pad_posterior=22.0)


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def normalize_stack(stack):
    """Clip every 2D image of stack (..., H, W) to its [1, 99] percentiles and rescale it to [0, 1]."""
    stack = stack.astype(np.float32)
    lead = stack.shape[:-2]
    flat = stack.reshape(-1, stack.shape[-2] * stack.shape[-1])
    p1, p99 = np.percentile(flat, (1, 99), axis=1)
    p1, p99 = p1.astype(np.float32)[:, None], p99.astype(np.float32)[:, None]
    flat = np.clip(flat, p1, p99)
    denom = np.where((p99 - p1) > 1e-8, p99 - p1, np.float32(1e-8))
    return ((flat - p1) / denom).reshape(lead + stack.shape[-2:])


def to_model_space_stack(stack, target_shape=TARGET_SHAPE):
    """Center-pad (zeros) and/or center-crop every 2D image of stack (..., H, W) to target_shape."""
    h, w = stack.shape[-2:]
    th, tw = target_shape
    pad_h, pad_w = max(th - h, 0), max(tw - w, 0)
    pad = [(0, 0)] * (stack.ndim - 2) + [(pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)]
    padded = np.pad(stack, pad, mode="constant")
    ph, pw = padded.shape[-2:]
    sh, sw = max((ph - th) // 2, 0), max((pw - tw) // 2, 0)
    return np.ascontiguousarray(padded[..., sh:sh + th, sw:sw + tw])


def compose_thetas(theta_a, theta_b):
    """Compose (..., 2, 3) affine_grid matrices: theta_a then theta_b."""
    a = np.asarray(theta_a, dtype=np.float64)
    b = np.asarray(theta_b, dtype=np.float64)
    bottom = np.broadcast_to(np.array([[0.0, 0.0, 1.0]]), a.shape[:-2] + (1, 3))
    Ma = np.concatenate([a, bottom], axis=-2)
    Mb = np.concatenate([b, bottom], axis=-2)
    return (Ma @ Mb)[..., :2, :].astype(np.float32)


def compute_reference(data, reference="average"):
    """Reference volume from data (..., T): temporal mean ("average") or first volume ("first")."""
    if reference == "average":
        return data.mean(axis=-1)
    elif reference == "first":
        return data[..., 0].copy()
    raise ValueError(f"Unknown reference strategy: {reference!r}")


def run_sc_crop(vol_rpi, affine_rpi, device):
    """
    Locate the spinal cord with sc_crop on a 3D volume (RPI array + its affine), as was done for the training data.
    Returns the inclusive voxel box (xmin, xmax, ymin, ymax, zmin, zmax) in RPI index space.
    """
    nii = nib.Nifti1Image(np.asarray(vol_rpi, dtype=np.float32), affine_rpi)
    try:
        bbox = sc_crop.detect(nii, device=str(device), **SC_CROP_PAD)
    except RuntimeError as e:
        raise RuntimeError(f"[moco-dl] sc_crop could not locate the spinal cord in the first volume: {e}") from e
    return tuple(int(bbox[k]) for k in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"))


# --------------------------------------------------------------------------
# Batched building blocks (numpy in / numpy out)
# --------------------------------------------------------------------------
def _chunks(n):
    for s in range(0, n, BATCH):
        yield slice(s, min(s + BATCH, n))


def predict_theta(model, moving_np, fixed_np, device):
    """moving_np, fixed_np: (N, H, W) -> theta (N, 2, 3). Chunked forward passes."""
    out = np.empty((moving_np.shape[0], 2, 3), dtype=np.float32)
    with torch.no_grad():
        for c in _chunks(moving_np.shape[0]):
            m = torch.from_numpy(np.ascontiguousarray(moving_np[c, None])).float().to(device)
            f = torch.from_numpy(np.ascontiguousarray(fixed_np[c, None])).float().to(device)
            _, affine_matrix = model(m, f)
            out[c] = affine_matrix.cpu().numpy()
    return out


def warp_batch_model_space(moving_np, theta_np, device, mode="bilinear"):
    """moving_np (N, H, W), theta_np (N, 2, 3) -> warped (N, H, W). Chunked."""
    out = np.empty(moving_np.shape, dtype=np.float32)
    with torch.no_grad():
        for c in _chunks(moving_np.shape[0]):
            m = torch.from_numpy(np.ascontiguousarray(moving_np[c, None])).float().to(device)
            th = torch.from_numpy(np.ascontiguousarray(theta_np[c])).float().to(device)
            out[c] = moco_dl_model.warp_with_affine(m, th, mode=mode)[:, 0].cpu().numpy()
    return out


def compute_ncc_batch(a, b):
    """Global NCC of N pairs of images (N, H, W) -> (N,) (population std)."""
    a = a.reshape(a.shape[0], -1).astype(np.float32)
    b = b.reshape(b.shape[0], -1).astype(np.float32)
    sa, sb = a.std(axis=1), b.std(axis=1)
    cov = ((a - a.mean(axis=1, keepdims=True)) * (b - b.mean(axis=1, keepdims=True))).mean(axis=1)
    ok = (sa >= 1e-8) & (sb >= 1e-8)
    return np.where(ok, cov / np.where(ok, sa * sb, 1.0), 0.0)


def non_regression_filter_batch(moving, fixed, theta_pred, device, tolerance=0.0):
    """Keep theta_pred where it does not lower the NCC by more than `tolerance`, else identity."""
    sim_before = compute_ncc_batch(moving, fixed)
    warped = warp_batch_model_space(moving, theta_pred, device, mode="bilinear")
    sim_after = compute_ncc_batch(warped, fixed)
    reject = sim_after < sim_before - tolerance
    out = theta_pred.copy()
    out[reject] = IDENTITY_THETA
    return out


# --------------------------------------------------------------------------
# Stage 2
# --------------------------------------------------------------------------
def refine_stage2(model2, moving_ms, theta_cum, fixed1, device, reference2="recompute",
                  max_iter=3, tol=1e-3, non_reg_tol=0.0):
    """
    Stage 2 refinement with early stopping (tolerance or divergence) and per-slice
    non-regression filter, batched over all (volume, slice) pairs.

    moving_ms: (T, Zc, H, W); theta_cum: (T, Zc, 2, 3); fixed1: (Zc, H, W).
    reference2: "recompute" (temporal mean of the corrected series at each iteration, as the
    code of infer_fmri.py does) or "same" (keep fixed1).
    """
    if reference2 not in ("recompute", "same"):
        raise ValueError(f"Unknown reference2 strategy: {reference2!r}")
    T, Zc, H, W = moving_ms.shape
    theta_cum = theta_cum.copy()

    def flat(a):
        return a.reshape((T * Zc,) + a.shape[2:])

    def warp_all(th):
        return warp_batch_model_space(flat(moving_ms), flat(th), device, mode="bilinear").reshape(T, Zc, H, W)

    current = warp_all(theta_cum)
    prev_mean_dev = float('inf')
    best_theta_cum = theta_cum.copy()

    for it in range(max_iter):
        if reference2 == "same":
            ref = fixed1
        else:
            ref = compute_reference(np.moveaxis(current, 0, -1), "average")
        ref_all = np.broadcast_to(ref[None], current.shape)
        res = predict_theta(model2, flat(current), flat(ref_all), device)
        res = non_regression_filter_batch(flat(current), flat(ref_all), res, device, tolerance=non_reg_tol)
        residuals = res.reshape(T, Zc, 2, 3)

        theta_cum = compose_thetas(theta_cum, residuals)
        current = warp_all(theta_cum)

        mean_dev = float(np.abs(residuals - IDENTITY_THETA).mean())
        printv(f"  Stage 2 iteration {it + 1}/{max_iter}: mean |residual - identity| = {mean_dev:.5f}")

        if mean_dev < tol:
            printv(f"  Converged (< tol={tol}), stopping early.")
            best_theta_cum = theta_cum.copy()
            break

        if it > 0 and mean_dev > prev_mean_dev:
            printv("  Residuals increasing (divergence), stopping early.")
            break

        prev_mean_dev = mean_dev
        best_theta_cum = theta_cum.copy()

    return best_theta_cum


def moco_dl(fname_data, fname_mask='', fname_ref='', path_out='', mode="fmri", fname_bvals='', fname_bvecs='',
            ckpt1_path=None, ckpt2_path=None):
    """
        Deep-learning motion correction (two-stage RigidRegistrationNet) for dMRI/fMRI.

        The signature is unchanged so that sct_fmri_moco / sct_dmri_moco keep working.
        fname_mask is not used: -m is rejected by check_dl_args(), the cord is located with sc_crop.

        ckpt1_path / ckpt2_path: optional checkpoint files, overriding the installed models
        ($SCT_DIR/data/moco-dl_models_{mode}/model1.pt and model2.pt). Used by the development entry point.

        Outputs (in path_out): <input>_mocoDL, its mean (fMRI) or b0/dwi means (dMRI), and
        moco_params_x/y.nii.gz: slice-wise translations (1, 1, Z, T) in mm, as physical LPS components
        (the convention of ITK/ANTs displacement fields).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(path_out, exist_ok=True)
    path_out_abs = os.path.abspath(path_out)

    # Installed models: one bundle per modality (sct_download_data -d moco-dl_models_fmri / moco-dl_models_dmri)
    models_dir = sct_dir_local_path("data", f"moco-dl_models_{mode}")
    ckpt1_path = ckpt1_path or os.path.join(models_dir, "model1.pt")
    ckpt2_path = ckpt2_path or os.path.join(models_dir, "model2.pt")

    if not os.path.exists(ckpt1_path) or not os.path.exists(ckpt2_path):
        raise FileNotFoundError(f"\n[moco-dl] MocoDL {mode} models not found. "
                                f"Please install them using: sct_download_data -d moco-dl_models_{mode}")

    printv("\n[INFO] Running DL-based motion correction...\n")
    # Display parameter summary
    printv(textwrap.dedent(f"""
        Input parameters (DL-based motion correction):
        ----------------------------------------------------
        Input file:            {fname_data}
        Output folder:         {path_out_abs}
        Mode:                  {mode}
        Reference image:       {fname_ref if fname_ref != '' else 'temporal mean of the input (recomputed in stage 2)'}
        Model 1:               {ckpt1_path}
        Model 2:               {ckpt2_path}
        bvecs (dmri only):     {fname_bvecs}
        bvals (dmri only):     {fname_bvals}
        Device:                {device}
    """))
    start_time = time.time()

    # Load native inputs
    im_data_native = Image(fname_data)
    im_ref_native = Image(fname_ref) if fname_ref else None
    orig_orient = im_data_native.orientation
    printv(f"\n[moco-dl] Native input orientation: {orig_orient}")

    # Validate orientation and number of volumes
    if orig_orient[2] in 'IS':
        printv("[moco-dl] Input is treated as axial (last axis = S/I).")
    else:
        raise ValueError(
            f"Unsupported orientation: {orig_orient}. "
            f"DL-based moco currently supports only inputs where the last axis is S/I."
        )
    if im_data_native.data.ndim != 4 or im_data_native.data.shape[3] < MIN_VOLUMES:
        raise ValueError(f"[moco-dl] Input must be 4D with at least {MIN_VOLUMES} volumes "
                         f"(got shape {im_data_native.data.shape}).")

    # Reorient inputs to RPI for inference
    printv("\n[moco-dl] Reorienting inputs to RPI for inference...")
    im_data_rpi = im_data_native.copy()
    if im_data_rpi.orientation != "RPI":
        im_data_rpi.change_orientation("RPI")
    # NB: change_orientation() updates the header but not the `affine` attribute, so read it from the header.
    affine_rpi = im_data_rpi.hdr.get_best_affine()
    data = np.asarray(im_data_rpi.data, dtype=np.float32)  # (X, Y, Z, T)
    X, Y, Z, T = data.shape

    ref_rpi = None
    if im_ref_native is not None:
        im_ref_rpi = im_ref_native.copy()
        if im_ref_rpi.orientation != "RPI":
            im_ref_rpi.change_orientation("RPI")
        ref_rpi = np.asarray(im_ref_rpi.data, dtype=np.float32)
        ref_rpi = ref_rpi[..., 0] if ref_rpi.ndim == 4 else ref_rpi
        if ref_rpi.shape != (X, Y, Z):
            raise ValueError(f"[moco-dl] Reference shape {ref_rpi.shape} does not match the input spatial shape {(X, Y, Z)}.")

    path_tmp = tmp_create(basename="moco-dl")
    try:
        fname_moco = _run(path_tmp, path_out_abs, fname_data, mode, fname_bvals, fname_bvecs, ckpt1_path, ckpt2_path,
                          device, start_time, im_data_native, im_data_rpi, affine_rpi, data, ref_rpi, orig_orient)
    finally:
        printv('\nDelete temporary files...')
        rmtree(path_tmp)

    printv(f"\n[moco-dl] Outputs saved in: {path_out_abs}")
    return fname_moco


def _run(path_tmp, path_out_abs, fname_data, mode, fname_bvals, fname_bvecs, ckpt1_path, ckpt2_path,
         device, start_time, im_data_native, im_data_rpi, affine_rpi, data, ref_rpi, orig_orient):
    """Crop, two-stage inference, final warp and outputs (everything that uses the tmp folder)."""
    X, Y, Z, T = data.shape

    # Crop around the spinal cord with sc_crop, on the first volume (box is inclusive)
    printv("\n[moco-dl] Locating the spinal cord with sc_crop (first volume)...")
    x0, x1, y0, y1, z0, z1 = run_sc_crop(data[..., 0], affine_rpi, device)
    x1, y1, z1 = x1 + 1, y1 + 1, z1 + 1
    Zc = z1 - z0
    printv(f"[moco-dl] sc_crop box (RPI voxels): x=[{x0}, {x1}), y=[{y0}, {y1}), z=[{z0}, {z1}) "
           f"-> {x1 - x0}x{y1 - y0}x{Zc} (network input {TARGET_SHAPE[0]}x{TARGET_SHAPE[1]})")
    if Zc < Z:
        printv(f"[moco-dl] {Z - Zc}/{Z} slices are outside the detected cord region and are left uncorrected.",
               type='warning')
    cropped = data[x0:x1, y0:y1, z0:z1]

    # Preprocessing: (X, Y, Z, T) -> (T, Z, X, Y), normalised per slice, padded/cropped to TARGET_SHAPE
    th, tw = TARGET_SHAPE
    moving_ms = to_model_space_stack(normalize_stack(np.transpose(cropped, (3, 2, 0, 1))))
    if ref_rpi is not None:
        reference_cropped, reference2 = ref_rpi[x0:x1, y0:y1, z0:z1], "same"
    else:
        reference_cropped, reference2 = compute_reference(cropped, "average"), "recompute"
    fixed1 = to_model_space_stack(normalize_stack(np.transpose(reference_cropped, (2, 0, 1))))

    # Load DL models
    printv(f"\n[moco-dl] Loading checkpoints: {ckpt1_path}, {ckpt2_path}")
    model1 = moco_dl_model.RigidRegistrationNet()
    model2 = moco_dl_model.RigidRegistrationNet()
    # The training scripts save plain state dicts (infer_fmri.py: model.load_state_dict(torch.load(...)))
    model1.load_state_dict(torch.load(ckpt1_path, map_location=device, weights_only=True))
    model2.load_state_dict(torch.load(ckpt2_path, map_location=device, weights_only=True))
    model1 = model1.to(device).eval()
    model2 = model2.to(device).eval()

    # Stage 1 + non-regression check (all (volume, slice) pairs in one batch)
    t0 = time.time()
    printv("\n[moco-dl] Running stage 1 with non-regression filter...")
    mov = moving_ms.reshape(T * Zc, th, tw)
    fix = np.broadcast_to(fixed1[None], moving_ms.shape).reshape(T * Zc, th, tw)
    pred = predict_theta(model1, mov, fix, device)
    theta1_all = non_regression_filter_batch(mov, fix, pred, device, tolerance=NON_REG_TOL).reshape(T, Zc, 2, 3)
    printv(f"[moco-dl] Stage 1 done in {time.time() - t0:.2f} s.")

    # Stage 2 + early stopping and non-regression check
    t0 = time.time()
    printv("\n[moco-dl] Running stage 2 with dynamic convergence & non-regression filter...")
    theta_total_all = refine_stage2(model2, moving_ms, theta1_all, fixed1, device, reference2=reference2,
                                    max_iter=N_ITER2, tol=TOL2, non_reg_tol=NON_REG_TOL)
    printv(f"[moco-dl] Stage 2 done in {time.time() - t0:.2f} s.")

    # The models only predict translations: theta = [[1, 0, tx], [0, 1, ty]] in normalized coordinates of the
    # TARGET_SHAPE canvas (H = x, W = y). As voxel shifts they do not depend on the canvas, so they apply directly
    # to the uncropped slices (this replaces transfer_theta_to_raw). Convention: out(x, y) = in(x + shift_x, y + shift_y).
    # Slices outside the sc_crop box keep a zero shift.
    shift_x = np.zeros((T, Z), dtype=np.float32)  # voxels along RPI x
    shift_y = np.zeros((T, Z), dtype=np.float32)  # voxels along RPI y
    shift_x[:, z0:z1] = theta_total_all[..., 1, 2] * (th / 2.0)
    shift_y[:, z0:z1] = theta_total_all[..., 0, 2] * (tw / 2.0)

    # Final warp in raw space (all (volume, slice) pairs of the crop's z range, batched)
    t0 = time.time()
    theta_raw = np.zeros((T, Zc, 2, 3), dtype=np.float32)
    theta_raw[..., 0, 0] = theta_raw[..., 1, 1] = 1.0
    theta_raw[..., 0, 2] = shift_y[:, z0:z1] * 2.0 / Y
    theta_raw[..., 1, 2] = shift_x[:, z0:z1] * 2.0 / X
    slices = np.transpose(data[:, :, z0:z1], (3, 2, 0, 1)).reshape(T * Zc, X, Y)
    warped_c = warp_batch_model_space(slices, theta_raw.reshape(T * Zc, 2, 3), device, mode=INTERP)
    warped = data.copy()
    warped[:, :, z0:z1] = np.transpose(warped_c.reshape(T, Zc, X, Y), (2, 3, 1, 0))
    printv(f"[moco-dl] Final warp ({INTERP}) done in {time.time() - t0:.2f} s.")
    printv(f"\nInference time: {time.time() - start_time:.2f} sec")

    # Write the outputs
    return _save_outputs(path_tmp, path_out_abs, fname_data, mode, fname_bvals, fname_bvecs,
                         warped, shift_x, shift_y, im_data_native, im_data_rpi, orig_orient)


def _save_outputs(path_tmp, path_out_abs, fname_data, mode, fname_bvals, fname_bvecs,
                  warped, shift_x, shift_y, im_data_native, im_data_rpi, orig_orient):
    """Write the corrected series, its mean(s) and the translation params; return the corrected series path."""
    affine_rpi = im_data_rpi.hdr.get_best_affine()

    # Save Moco output, back to native orientation
    im_moco = Image(np.ascontiguousarray(warped), hdr=im_data_rpi.hdr.copy())
    im_moco.affine = affine_rpi
    im_moco.hdr.set_data_shape(im_moco.data.shape)
    im_moco.hdr.set_data_dtype(np.float32)
    if orig_orient != "RPI":
        printv(f"\n[moco-dl] Reorienting corrected image from RPI back to native orientation: {orig_orient}")
        im_moco.change_orientation(orig_orient)
    fname_moco_tmp = os.path.join(path_tmp, "mocoDL.nii.gz")
    im_moco.save(fname_moco_tmp)

    # Save mean output
    printv("\n[moco-dl] Computing time-averaged output volume...")
    if mode == "dmri":
        args = ['-i', fname_moco_tmp, '-bvec', os.path.abspath(fname_bvecs), '-a', '1', '-ofolder', path_tmp, '-v', '0']
        if fname_bvals:
            args.extend(['-bval', os.path.abspath(fname_bvals)])
        _, fname_b0_mean, _, fname_dwi_mean = sct_dmri_separate_b0_and_dwi.main(argv=args)
    else:
        fname_moco_mean = add_suffix(fname_moco_tmp, '_mean')
        im_moco.mean(dim=3).save(fname_moco_mean)

    # Translation params in mm, as physical LPS components (ITK/ANTs displacement-field convention).
    # SCT's RPI is LAS in RAS terms: +x(RPI) = +L = +x(LPS), +y(RPI) = +A = -y(LPS), hence the sign flip of Ty.
    sx, sy = (float(np.linalg.norm(affine_rpi[:3, i])) for i in (0, 1))
    printv(f"\n[moco-dl] Scaling by voxel size from affine: sx={sx:.3f}mm, sy={sy:.3f}mm")
    params = {"x": (shift_x * sx).T, "y": (-shift_y * sy).T}  # (Z, T)
    fname_params_tmp = {}
    printv("\n[moco-dl] Saving translation params...")
    for axis, values in params.items():
        # Built in RPI then reoriented like the data, so that the slice axis follows the native slice order.
        # The values are physical vector components, so they are not changed by the reorientation.
        im_p = Image(np.ascontiguousarray(values[np.newaxis, np.newaxis].astype(np.float32)), hdr=im_data_rpi.hdr.copy())
        im_p.affine = affine_rpi
        im_p.hdr.set_data_shape(im_p.data.shape)  # (1, 1, Z, T)
        im_p.hdr.set_data_dtype(np.float32)
        if orig_orient != "RPI":
            im_p.change_orientation(orig_orient)
        im_p.affine = im_data_native.hdr.get_best_affine()
        fname_params_tmp[axis] = os.path.join(path_tmp, f"moco_params_{axis}.nii.gz")
        im_p.save(fname_params_tmp[axis])

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
    for axis in ("x", "y"):
        generate_output_file(fname_params_tmp[axis], os.path.join(path_out_abs, f"moco_params_{axis}.nii.gz"),
                             squeeze_data=False)
    return fname_moco
