__author__ = 'slevy'


import numpy as np
import nibabel as nib
import pylab
import matplotlib.legend_handler as lgd
from sct_extract_metric import extract_metric_within_tract

subject='hc_sc_003'

fname_PD_map = '/home/django/slevy/data/boston/'+subject+'/mtv/PD_map.nii.gz'
fname_T1_map = '/home/django/slevy/data/boston/'+subject+'/mtv/T1_map.nii.gz'
fname_MTVF_map = '/home/django/slevy/data/boston/'+subject+'/mtv/mtvf_map.nii.gz'
fname_SPGR10_map = '/home/django/slevy/data/boston/'+subject+'/mtv/spgr10_crop.nii.gz'
fname_CSF_mask = '/home/django/slevy/data/boston/'+subject+'/mtv/spgr10_crop_csf_mask.nii.gz'
fname_cord_mask = '/home/django/slevy/data/boston/'+subject+'/mtv/spgr10_crop_seg.nii.gz'

PD_map = nib.load(fname_PD_map).get_data()
T1_map = nib.load(fname_T1_map).get_data()
MTVF_map = nib.load(fname_MTVF_map).get_data()
SPGR10 = nib.load(fname_SPGR10_map).get_data()
#SPGR10 = np.float64(SPGR10)
CSF_mask = nib.load(fname_CSF_mask).get_data()
cord_mask = nib.load(fname_cord_mask).get_data()

(nx, ny, nz) = PD_map.shape

PD_mean_in_CSF_per_slice = np.empty((nz, 2))
T1_mean_in_CSF_per_slice = np.empty((nz, 2))
MTVF_mean_in_CSF_per_slice = np.empty((nz, 2))
SPGR10_mean_in_CSF_per_slice = np.empty((nz, 2))

PD_mean_in_cord_per_slice = np.empty((nz, 2))
T1_mean_in_cord_per_slice = np.empty((nz, 2))
MTVF_mean_in_cord_per_slice = np.empty((nz, 2))
SPGR10_mean_in_cord_per_slice = np.empty((nz, 2))

for z in range(0, nz):
    CSF_mask_slice = np.empty((1), dtype=object)
    cord_mask_slice = np.empty((1), dtype=object)

    CSF_mask_slice[0] = CSF_mask[..., z]
    cord_mask_slice[0] = cord_mask[..., z]

    PD_mean_in_CSF_per_slice[z, :] = extract_metric_within_tract(PD_map[..., z], CSF_mask_slice, 'wa', 1)
    PD_mean_in_cord_per_slice[z, :] = extract_metric_within_tract(PD_map[..., z], cord_mask_slice, 'wa', 1)

    T1_mean_in_CSF_per_slice[z, :] = extract_metric_within_tract(T1_map[..., z], CSF_mask_slice, 'wa', 1)
    T1_mean_in_cord_per_slice[z, :] = extract_metric_within_tract(T1_map[..., z], cord_mask_slice, 'wa', 1)

    MTVF_mean_in_CSF_per_slice[z, :] = extract_metric_within_tract(MTVF_map[..., z], CSF_mask_slice, 'wa', 1)
    MTVF_mean_in_cord_per_slice[z, :] = extract_metric_within_tract(MTVF_map[..., z], cord_mask_slice, 'wa', 1)

    SPGR10_mean_in_CSF_per_slice[z, :] = extract_metric_within_tract(SPGR10[..., z], CSF_mask_slice, 'wa', 1)
    SPGR10_mean_in_cord_per_slice[z, :] = extract_metric_within_tract(SPGR10[..., z], cord_mask_slice, 'wa', 1)

# ------------------------------PLOTS-----------------------------------------------------------------------------------
fig1 = pylab.figure(1)
fig1.suptitle('Mean T1, PD, MTVF and SPGR signal with flip angle=10deg in CSF and cord slice-by-slice', fontsize=20)

T1_ax = fig1.add_subplot(221, title='T1')
T1_ax.grid(True)
T1_ax.set_xlabel('Slices', fontsize=18)
T1_ax.set_ylabel('Mean T1 +/- std', fontsize=18)
T1_ax.errorbar(range(0, nz), T1_mean_in_CSF_per_slice[:, 0], T1_mean_in_CSF_per_slice[:, 1], marker='o')  # CSF
T1_ax.errorbar(range(0, nz), T1_mean_in_cord_per_slice[:, 0], T1_mean_in_cord_per_slice[:, 1], marker='o', color='r')  # cord
T1_ax.legend(['CSF', 'cord'], loc=2, handler_map={lgd.Line2D: lgd.HandlerLine2D(numpoints=1)}, fontsize=18)

PD_ax = fig1.add_subplot(222, title='PD')
PD_ax.grid(True)
PD_ax.set_xlabel('Slices', fontsize=18)
PD_ax.set_ylabel('Mean PD +/- std', fontsize=18)
PD_ax.errorbar(range(0, nz), PD_mean_in_CSF_per_slice[:, 0], PD_mean_in_CSF_per_slice[:, 1], marker='o')  # CSF
PD_ax.errorbar(range(0, nz), PD_mean_in_cord_per_slice[:, 0], PD_mean_in_cord_per_slice[:, 1], marker='o', color='r')  # cord
PD_ax.legend(['CSF', 'cord'], loc=2, handler_map={lgd.Line2D: lgd.HandlerLine2D(numpoints=1)}, fontsize=18)

MTVF_ax = fig1.add_subplot(223, title='MTVF')
MTVF_ax.grid(True)
MTVF_ax.set_xlabel('Slices', fontsize=18)
MTVF_ax.set_ylabel('Mean MTVF +/- std', fontsize=18)
MTVF_ax.errorbar(range(0, nz), MTVF_mean_in_CSF_per_slice[:, 0], MTVF_mean_in_CSF_per_slice[:, 1], marker='o')  # CSF
MTVF_ax.errorbar(range(0, nz), MTVF_mean_in_cord_per_slice[:, 0], MTVF_mean_in_cord_per_slice[:, 1], marker='o', color='r')  # cord
MTVF_ax.legend(['CSF', 'cord'], loc=2, handler_map={lgd.Line2D: lgd.HandlerLine2D(numpoints=1)}, fontsize=18)

SPGR10_ax = fig1.add_subplot(224, title='SPGR signal with flip angle=10deg')
SPGR10_ax.grid(True)
SPGR10_ax.set_xlabel('Slices', fontsize=18)
SPGR10_ax.set_ylabel('Mean signal +/- std in SPGR image with flip angle=10deg', fontsize=18)
SPGR10_ax.errorbar(range(0, nz), SPGR10_mean_in_CSF_per_slice[:, 0], SPGR10_mean_in_CSF_per_slice[:, 1], marker='o')  # CSF
SPGR10_ax.errorbar(range(0, nz), SPGR10_mean_in_cord_per_slice[:, 0], SPGR10_mean_in_cord_per_slice[:, 1], marker='o', color='r')  # cord
SPGR10_ax.legend(['CSF', 'cord'], loc=2, handler_map={lgd.Line2D: lgd.HandlerLine2D(numpoints=1)}, fontsize=18)

pylab.show()