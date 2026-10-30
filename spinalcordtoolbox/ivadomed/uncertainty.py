import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import label, generate_binary_structure
from pathlib import Path
import json
import numpy as np
from ivadomed import postprocessing as imed_postpro
from typing import List


def run_uncertainty(image_folder):
    """Compute uncertainty from model prediction.

    This function loops across the model predictions (nifti masks) and estimates the uncertainty from the Monte Carlo
    samples. Both voxel-wise and structure-wise uncertainty are estimates.

    Args:
        image_folder (str): Folder containing the Monte Carlo samples.
    """
    # list subj_acq prefixes
    subj_acq_lst = [file.name.split('_pred')[0] for file in Path(image_folder).iterdir()
                    if file.name.endswith('.nii.gz') and '_pred' in file.name]
    # remove duplicates
    subj_acq_lst = list(set(subj_acq_lst))
    # keep only the images where unc has not been computed yet
    subj_acq_lst = [file for file in subj_acq_lst if not Path(image_folder, file + '_unc-cv.nii.gz').is_file()]

    # loop across subj_acq
    for subj_acq in tqdm(subj_acq_lst, desc="Uncertainty Computation"):
        # hard segmentation from MC samples
        fname_pred: Path = Path(image_folder, subj_acq + '_pred.nii.gz')
        # fname for soft segmentation from MC simulations
        fname_soft: Path = Path(image_folder, subj_acq + '_soft.nii.gz')
        # find Monte Carlo simulations
        fname_pred_lst: List[str] = []
        for file in Path(image_folder).iterdir():
            if subj_acq + '_pred_' in file.name and ('_painted' not in file.name) and ('_color' not in file.name):
                fname_pred_lst.append(str(file))

        # if final segmentation from Monte Carlo simulations has not been generated yet
        if not fname_pred.is_file() or not fname_soft.is_file():
            # threshold used for the hard segmentation
            thr = 1. / len(fname_pred_lst)  # 1 for all voxels where at least on MC sample predicted 1
            # average then argmax
            combine_predictions(fname_pred_lst, str(fname_pred), str(fname_soft), thr=thr)

        fname_unc_vox = Path(image_folder, subj_acq + '_unc-vox.nii.gz')
        if not fname_unc_vox.is_file():
            # compute voxel-wise uncertainty map
            voxelwise_uncertainty(fname_pred_lst, str(fname_unc_vox))

        fname_unc_struct = Path(image_folder, subj_acq + '_unc.nii.gz')
        if not Path(image_folder, subj_acq + '_unc-cv.nii.gz').is_file():
            # compute structure-wise uncertainty
            structurewise_uncertainty(fname_pred_lst, str(fname_pred), str(fname_unc_vox), str(fname_unc_struct))


def combine_predictions(fname_lst, fname_hard, fname_prob, thr=0.5):
    """Combine predictions from Monte Carlo simulations.

    Combine predictions from Monte Carlo simulations and save the resulting as:
        (1) `fname_prob`, a soft segmentation obtained by averaging the Monte Carlo samples.
        (2) `fname_hard`, a hard segmentation obtained thresholding with `thr`.

    Args:
        fname_lst (list of str): List of the Monte Carlo samples.
        fname_hard (str): Filename for the output hard segmentation.
        fname_prob (str): Filename for the output soft segmentation.
        thr (float): Between 0 and 1. Used to threshold the soft segmentation and generate the hard segmentation.
    """
    # collect all MC simulations
    mc_data = np.array([nib.load(fname).get_fdata() for fname in fname_lst])
    first_file_header = nib.load(fname_lst[0]).header

    # average over all the MC simulations
    data_prob = np.mean(mc_data, axis=0)
    # save prob segmentation
    nib_prob = nib.Nifti1Image(
        dataobj=data_prob,
        affine=first_file_header.get_best_affine(),
        header=first_file_header.copy()
    )
    nib.save(nib_prob, fname_prob)

    # argmax operator
    data_hard = imed_postpro.threshold_predictions(data_prob, thr=thr).astype(np.uint8)
    # save hard segmentation
    nib_hard = nib.Nifti1Image(
        dataobj=data_hard,
        affine=first_file_header.get_best_affine(),
        header=first_file_header.copy()
    )
    nib.save(nib_hard, fname_hard)


def voxelwise_uncertainty(fname_lst, fname_out, eps=1e-5):
    """Estimate voxel wise uncertainty.

    Voxel-wise uncertainty is estimated as entropy over all N MC probability maps, and saved in `fname_out`.

    Args:
        fname_lst (list of str): List of the Monte Carlo samples.
        fname_out (str): Output filename.
        eps (float): Epsilon value to deal with np.log(0).
    """
    # collect all MC simulations
    mc_data = np.array([nib.load(fname).get_fdata() for fname in fname_lst])
    affine = nib.load(fname_lst[0]).header.get_best_affine()

    # entropy
    unc = np.repeat(np.expand_dims(mc_data, -1), 2, -1)  # n_it, x, y, z, 2
    unc[..., 0] = 1 - unc[..., 1]
    unc = -np.sum(np.mean(unc, 0) * np.log(np.mean(unc, 0) + eps), -1)

    # Clip values to 0
    unc[unc < 0] = 0

    # save uncertainty map
    nib_unc = nib.Nifti1Image(unc, affine)
    nib.save(nib_unc, fname_out)


def structurewise_uncertainty(fname_lst, fname_hard, fname_unc_vox, fname_out):
    """Estimate structure wise uncertainty.

    Structure-wise uncertainty from N MC probability maps (`fname_lst`) and saved in `fname_out` with the following
    suffixes:

        * '-cv.nii.gz': coefficient of variation
        * '-iou.nii.gz': intersection over union
        * '-avgUnc.nii.gz': average voxel-wise uncertainty within the structure.

    Args:
        fname_lst (list of str): List of the Monte Carlo samples.
        fname_hard (str): Filename of the hard segmentation, which is used to compute the `avgUnc` by providing a mask
            of the structures.
        fname_unc_vox (str): Filename of the voxel-wise uncertainty, which is used to compute the `avgUnc`.
        fname_out (str): Output filename.
    """
    # 18-connectivity
    bin_struct = np.array(generate_binary_structure(3, 2))

    # load hard segmentation
    nib_hard = nib.load(fname_hard)
    data_hard = nib_hard.get_fdata()
    # Label each object of each class
    data_hard_labeled = [label(data_hard[..., i_class], structure=bin_struct)[0] for i_class in
                         range(data_hard.shape[-1])]

    # load all MC simulations (in mc_dict["mc_data"]) and label them (in mc_dict["mc_labeled"])
    mc_dict = {"mc_data": [], "mc_labeled": []}
    for fname in fname_lst:
        data = nib.load(fname).get_fdata()
        mc_dict["mc_data"].append([data[..., i_class] for i_class in range(data.shape[-1])])

        labeled_list = [label(data[..., i_class], structure=bin_struct)[0] for i_class in range(data.shape[-1])]
        mc_dict["mc_labeled"].append(labeled_list)

    # load uncertainty map
    data_uncVox = nib.load(fname_unc_vox).get_fdata()

    # Init output arrays
    data_iou, data_cv, data_avgUnc = np.zeros(data_hard.shape), np.zeros(data_hard.shape), np.zeros(data_hard.shape)

    # Loop across classes
    for i_class in range(data_hard.shape[-1]):
        # Hard segmentation of the i_class that has been labeled
        data_hard_labeled_class = data_hard_labeled[i_class]
        # Get number of objects in
        l, l_count = np.unique(data_hard_labeled_class, return_counts=True)

        # Get all non zero labels and exclude structure of 1 pixel
        labels = l[l_count != 1][1:]
        # Loop across objects
        for i_obj in labels:
            # select the current structure, remaining voxels are set to zero
            data_hard_labeled_class_obj = (np.array(data_hard_labeled_class) == i_obj).astype(int)

            # Get object coordinates
            xx_obj, yy_obj, zz_obj = np.where(data_hard_labeled_class_obj)

            # Loop across the MC samples and mask the structure of interest
            data_class_obj_mc = []
            for i_mc in range(len(fname_lst)):
                # Get index of the structure of interest in the MC sample labeled
                i_mc_labels, i_mc_counts = np.unique(data_hard_labeled_class_obj * mc_dict["mc_labeled"][i_mc][i_class],
                                                     return_counts=True)
                i_mc_label = i_mc_labels[np.argmax(i_mc_counts[1:]) + 1] if len(i_mc_counts) > 1 else 0

                data_tmp = np.zeros(mc_dict["mc_data"][i_mc][i_class].shape)
                # If i_mc_label is zero, it means the structure is not present in this mc_sample
                if i_mc_label > 0:
                    data_tmp[mc_dict["mc_labeled"][i_mc][i_class] == i_mc_label] = 1.

                data_class_obj_mc.append(data_tmp.astype(bool))

            # COMPUTE IoU
            # Init intersection and union
            intersection = np.logical_and(data_class_obj_mc[0], data_class_obj_mc[1])
            union = np.logical_or(data_class_obj_mc[0], data_class_obj_mc[1])
            # Loop across remaining MC samples
            for i_mc in range(2, len(data_class_obj_mc)):
                intersection = np.logical_and(intersection, data_class_obj_mc[i_mc])
                union = np.logical_or(union, data_class_obj_mc[i_mc])
            # Compute float
            iou = np.sum(intersection) * 1. / np.sum(union)
            # assign uncertainty value to the structure
            data_iou[xx_obj, yy_obj, zz_obj, i_class] = iou

            # COMPUTE COEFFICIENT OF VARIATION
            # List of volumes for each MC sample
            vol_mc_lst = [np.sum(data_class_obj_mc[i_mc]) for i_mc in range(len(data_class_obj_mc))]
            # Mean volume
            mu_mc = np.mean(vol_mc_lst)
            # STD volume
            sigma_mc = np.std(vol_mc_lst)
            # Coefficient of variation
            cv = sigma_mc / mu_mc
            # assign uncertainty value to the structure
            data_cv[xx_obj, yy_obj, zz_obj, i_class] = cv

            # COMPUTE AVG VOXEL WISE UNC
            avgUnc = np.mean(data_uncVox[xx_obj, yy_obj, zz_obj, i_class])
            # assign uncertainty value to the structure
            data_avgUnc[xx_obj, yy_obj, zz_obj, i_class] = avgUnc

    # save nifti files
    fname_iou = fname_out.split('.nii.gz')[0] + '-iou.nii.gz'
    fname_cv = fname_out.split('.nii.gz')[0] + '-cv.nii.gz'
    fname_avgUnc = fname_out.split('.nii.gz')[0] + '-avgUnc.nii.gz'

    nib_iou = nib.Nifti1Image(
        dataobj=data_iou,
        affine=nib_hard.header.get_best_affine(),
        header=nib_hard.header.copy()
    )
    nib_cv = nib.Nifti1Image(
        dataobj=data_cv,
        affine=nib_hard.header.get_best_affine(),
        header=nib_hard.header.copy()
    )
    nib_avgUnc = nib.Nifti1Image(
        data_avgUnc,
        affine=nib_hard.header.get_best_affine(),
        header=nib_hard.header.copy()
    )

    nib.save(nib_iou, fname_iou)
    nib.save(nib_cv, fname_cv)
    nib.save(nib_avgUnc, fname_avgUnc)
