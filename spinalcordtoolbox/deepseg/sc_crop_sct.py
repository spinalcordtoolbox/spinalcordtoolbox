import nibabel as nib
import numpy as np


def uncrop(seg_nii, bbox) -> "nib.Nifti1Image":
    
    original_img = bbox["_original_img"]
    xmin, xmax   = bbox["xmin"], bbox["xmax"]
    ymin, ymax   = bbox["ymin"], bbox["ymax"]
    zmin, zmax   = bbox["zmin"], bbox["zmax"]

    dtype   = seg_nii.get_data_dtype()
    full    = np.zeros(original_img.shape[:3], dtype=dtype)
    seg_arr = np.asarray(seg_nii.dataobj).astype(dtype)
    full[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1] = seg_arr

    return nib.Nifti1Image(full, original_img.affine, original_img.header)