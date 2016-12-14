#!/bin/bash
# batch to debug PVE stuff
# author: Julien Cohen-Adad
# 2014-11-27

sct_extract_metric -i WM_phantom_noise.nii.gz -f label_tracts -m ml
# --> completely off!

# created inverted mask from WM
fslmaths WMtract__all.nii.gz -add -1 -mul -1 mask_inv

sct_extract_metric -i WM_phantom_noise.nii.gz -f label_tracts_inv -m ml
# --> completely off too!

# when including all voxels in estimation (including those almost equal to 0)
sct_extract_metric -i WM_phantom_noise.nii.gz -f label_tracts -m ml -o metric_label_tracts_allVox.txt
# --> off

# same thing with inverted mask
sct_extract_metric -i WM_phantom_noise.nii.gz -f label_tracts_inv -m ml -o metric_label_tracts_inv_allVox.txt
# --> very similar to metric_label_tracts_allVox

# check with only two classes
sct_extract_metric -i WM_phantom_noise.nii.gz -f label_wm_inv -m ml -o metric_label_wm_inv.txt
# --> ok

# check with single value input within WM
sct_extract_metric -i WMtract__all.nii.gz -f label_wm_inv -m ml -o metric_WMtracts_label_wm_inv.txt
# --> ok, perfect:
# [  1.00000000e+00  -4.05434692e-19]

# rewrote ML estimation in sct_extract_metrics:
# beta, residuals, rank, singular_value = np.linalg.lstsq(x, y, rcond=-1)
# instead of:
# beta, residuals, rank, singular_value = np.linalg.lstsq(np.dot(x.T, x), np.dot(x.T, y), rcond=-1)
sct_extract_metric -i WMtract__all.nii.gz -f label_wm_inv -m ml -o metric_WMtracts_label_wm_inv.txt
# better solution, and now residuals are not empty:
# [  1.00000000e+00  -1.04086030e-18] [  9.84284275e-28] 2 [ 266.69560352   25.53021159]

sct_extract_metric -i WM_phantom_noise.nii.gz -f label_wm_inv -m ml -o metric_label_wm_inv.txt
# [  4.08465452e+01  -7.06262464e-04] [ 3713.39401725] 2 [ 266.69560352   25.53021159]

# now back with multiple tracts:
sct_extract_metric -i WM_phantom_noise.nii.gz -f label_tracts -m ml -o metric_label_tracts_allVox.txt
# --> bad...

# now with inverted:
sct_extract_metric -i WM_phantom_noise.nii.gz -f label_tracts_inv -m ml -o metric_label_tracts_inv_allVox.txt
# also bad...

# maybe need to have X square and of full rank. trying this...

