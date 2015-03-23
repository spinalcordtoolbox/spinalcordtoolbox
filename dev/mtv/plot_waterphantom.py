__author__ = 'slevy_local'

import nibabel as nib
import os
from sct_plot_tools import *
import matplotlib.pyplot as plt

# ================================ PARAMETERS ==========================================================================
working_dir = '/Volumes/slevy-3/data/criugm/water_phantom/mtv_tr20'
fname_mask = 'mask.nii.gz'
fname_pd = 'PD_map.nii.gz'
fname_t1 = 'T1_map.nii.gz'

# ======================================================================================================================
os.chdir(working_dir)

# load data
pd = nib.load(fname_pd).get_data()
t1 = nib.load(fname_t1).get_data()
mask = nib.load(fname_mask).get_data()

# compute metric slice by slice
nz = pd.shape[2]
pd_mean_std = compute_metric_mean_and_std_slice_by_slice(pd, mask)
t1_mean_std = compute_metric_mean_and_std_slice_by_slice(t1, mask)

# plot
ax_pd = plt.subplot(121)
ax_pd.errorbar(range(0, nz), pd_mean_std[:, 0], pd_mean_std[:, 1], label='PD')
# ax_pd.set_ylim([12000, 14500])
ax_pd.legend()
ax_pd.set_xlabel('z')

ax_t1 = plt.subplot(122)
ax_t1.errorbar(range(0, nz), t1_mean_std[:, 0], t1_mean_std[:, 1], label='T1')
ax_t1.set_ylim([6.28, 6.29])
ax_t1.legend()
ax_t1.set_xlabel('z')

plt.savefig('plot.pdf')

plt.show()

