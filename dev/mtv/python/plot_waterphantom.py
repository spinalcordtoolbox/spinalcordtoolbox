__author__ = 'slevy_local'

import nibabel as nib
import os
from sct_tools import *
import matplotlib.pyplot as plt

# ================================ PARAMETERS ==========================================================================
working_dir = '/Volumes/slevy-3/data/criugm/water_phantom'
# B1
fname_b1_no_smooth = 'b1/b1_scaling_no_smooth_mask_cropped.nii.gz'
fname_b1_smooth = 'b1/b1_scaling_smooth_mask_cropped.nii.gz'
# TR=20 ms
fname_pd_noB1_tr20 = 'mtv_tr20/PD_map_no_b1.nii.gz'
fname_t1_noB1_tr20 = 'mtv_tr20/T1_map_no_b1.nii.gz'
fname_pd_noB1_tr20m = 'mtv_tr20/PD_map_no_b1_matlab.nii.gz'  # matlab
fname_t1_noB1_tr20m = 'mtv_tr20/T1_map_no_b1_matlab.nii.gz'  # matlab
fname_pd_B1_tr20 = 'mtv_tr20/PD_map_b1_corr.nii.gz'
fname_t1_B1_tr20 = 'mtv_tr20/T1_map_b1_corr.nii.gz'
fname_pd_B1_tr20m = 'mtv_tr20/PD_map_b1_corr_matlab.nii.gz'  # matlab
fname_t1_B1_tr20m = 'mtv_tr20/T1_map_b1_corr_matlab.nii.gz'  # matlab
fname_spgr5_tr20 = 'mtv_tr20/spgr5mask_cropped.nii.gz'
fname_spgr10_tr20 = 'mtv_tr20/spgr10mask_cropped.nii.gz'
fname_spgr20_tr20 = 'mtv_tr20/spgr20mask_cropped.nii.gz'
fname_spgr30_tr20 = 'mtv_tr20/spgr30mask_cropped.nii.gz'
# TR=10 ms
fname_pd_noB1_tr10 = 'mtv_tr10/PD_map_no_b1.nii.gz'
fname_t1_noB1_tr10 = 'mtv_tr10/T1_map_no_b1.nii.gz'
fname_pd_B1_tr10 = 'mtv_tr10/PD_map_b1_corr.nii.gz'
fname_t1_B1_tr10 = 'mtv_tr10/T1_map_b1_corr.nii.gz'
fname_spgr5_tr10 = 'mtv_tr10/spgr5mask_cropped.nii.gz'
fname_spgr10_tr10 = 'mtv_tr10/spgr10mask_cropped.nii.gz'
fname_spgr20_tr10 = 'mtv_tr10/spgr20mask_cropped.nii.gz'
fname_spgr30_tr10 = 'mtv_tr10/spgr30mask_cropped.nii.gz'


# ======================================================================================================================
os.chdir(working_dir)

# load data
# B1
b1_no_smooth = nib.load(fname_b1_no_smooth).get_data()
b1_smooth = nib.load(fname_b1_smooth).get_data()
# TR=20 ms
pd_noB1_tr20 = nib.load(fname_pd_noB1_tr20).get_data()
t1_noB1_tr20 = nib.load(fname_t1_noB1_tr20).get_data()
pd_noB1_tr20m = nib.load(fname_pd_noB1_tr20m).get_data()  # matlab
t1_noB1_tr20m = nib.load(fname_t1_noB1_tr20m).get_data()  # matlab
pd_B1_tr20 = nib.load(fname_pd_B1_tr20).get_data()
t1_B1_tr20 = nib.load(fname_t1_B1_tr20).get_data()
pd_B1_tr20m = nib.load(fname_pd_B1_tr20m).get_data()  # matlab
t1_B1_tr20m = nib.load(fname_t1_B1_tr20m).get_data()  # matlab
spgr5_tr20 = nib.load(fname_spgr5_tr20).get_data()
spgr10_tr20 = nib.load(fname_spgr10_tr20).get_data()
spgr20_tr20 = nib.load(fname_spgr20_tr20).get_data()
spgr30_tr20 = nib.load(fname_spgr30_tr20).get_data()
# TR=10 ms
pd_noB1_tr10 = nib.load(fname_pd_noB1_tr10).get_data()
t1_noB1_tr10 = nib.load(fname_t1_noB1_tr10).get_data()
pd_B1_tr10 = nib.load(fname_pd_B1_tr10).get_data()
t1_B1_tr10 = nib.load(fname_t1_B1_tr10).get_data()
spgr5_tr10 = nib.load(fname_spgr5_tr10).get_data()
spgr10_tr10 = nib.load(fname_spgr10_tr10).get_data()
spgr20_tr10 = nib.load(fname_spgr20_tr10).get_data()
spgr30_tr10 = nib.load(fname_spgr30_tr10).get_data()


# compute metric slice by slice
# B1
b1_no_smooth_mean_std = compute_metric_mean_and_std_slice_by_slice(b1_no_smooth)
b1_smooth_mean_std = compute_metric_mean_and_std_slice_by_slice(b1_smooth)
# TR=20 ms
nz_tr20 = pd_noB1_tr20.shape[2]
pd_noB1_mean_std_tr20 = compute_metric_mean_and_std_slice_by_slice(pd_noB1_tr20)
t1_noB1_mean_std_tr20 = compute_metric_mean_and_std_slice_by_slice(t1_noB1_tr20)
pd_noB1_mean_std_tr20m = compute_metric_mean_and_std_slice_by_slice(pd_noB1_tr20m)  # matlab
t1_noB1_mean_std_tr20m = compute_metric_mean_and_std_slice_by_slice(t1_noB1_tr20m)  # matlab
pd_B1_mean_std_tr20 = compute_metric_mean_and_std_slice_by_slice(pd_B1_tr20)
t1_B1_mean_std_tr20 = compute_metric_mean_and_std_slice_by_slice(t1_B1_tr20)
pd_B1_mean_std_tr20m = compute_metric_mean_and_std_slice_by_slice(pd_B1_tr20m)  # matlab
t1_B1_mean_std_tr20m = compute_metric_mean_and_std_slice_by_slice(t1_B1_tr20m)  # matlab
spgr5_mean_std_tr20 = compute_metric_mean_and_std_slice_by_slice(spgr5_tr20)
spgr10_mean_std_tr20 = compute_metric_mean_and_std_slice_by_slice(spgr10_tr20)
spgr20_mean_std_tr20 = compute_metric_mean_and_std_slice_by_slice(spgr20_tr20)
spgr30_mean_std_tr20 = compute_metric_mean_and_std_slice_by_slice(spgr30_tr20)
# TR=10 ms
nz_tr10 = pd_noB1_tr10.shape[2]
pd_noB1_mean_std_tr10 = compute_metric_mean_and_std_slice_by_slice(pd_noB1_tr10)
t1_noB1_mean_std_tr10 = compute_metric_mean_and_std_slice_by_slice(t1_noB1_tr10)
pd_B1_mean_std_tr10 = compute_metric_mean_and_std_slice_by_slice(pd_B1_tr10)
t1_B1_mean_std_tr10 = compute_metric_mean_and_std_slice_by_slice(t1_B1_tr10)
spgr5_mean_std_tr10 = compute_metric_mean_and_std_slice_by_slice(spgr5_tr10)
spgr10_mean_std_tr10 = compute_metric_mean_and_std_slice_by_slice(spgr10_tr10)
spgr20_mean_std_tr10 = compute_metric_mean_and_std_slice_by_slice(spgr20_tr10)
spgr30_mean_std_tr10 = compute_metric_mean_and_std_slice_by_slice(spgr30_tr10)


# plot
fig1 = plt.figure(figsize=(41, 22))
ax_pd = fig1.add_subplot(221)
# TR=20 ms
ax_pd.errorbar(range(0, nz_tr20), pd_noB1_mean_std_tr20[:, 0], pd_noB1_mean_std_tr20[:, 1], label='PD without B1 correction (TR=20ms)', color='b')
ax_pd.errorbar(range(0, nz_tr20), pd_B1_mean_std_tr20[:, 0], pd_B1_mean_std_tr20[:, 1], label='PD with B1 correction (TR=20ms)', color='g')
ax_pd.errorbar(range(0, nz_tr20), pd_noB1_mean_std_tr20m[:, 0], pd_noB1_mean_std_tr20m[:, 1], label='PD without B1 correction (TR=20ms)-matlab', color='b', lw=3.0)  # matlab
ax_pd.errorbar(range(0, nz_tr20), pd_B1_mean_std_tr20m[:, 0], pd_B1_mean_std_tr20m[:, 1], label='PD with B1 correction (TR=20ms)-matlab', color='g', lw=3.0)  # matlab
# TR=10 ms
ax_pd.errorbar(range(0, nz_tr10), pd_noB1_mean_std_tr10[:, 0], pd_noB1_mean_std_tr10[:, 1], label='PD without B1 correction (TR=10ms)', color='b', ls='--')
ax_pd.errorbar(range(0, nz_tr10), pd_B1_mean_std_tr10[:, 0], pd_B1_mean_std_tr10[:, 1], label='PD with B1 correction (TR=10ms)', color='g', ls='--')

ax_pd.set_ylim([8000, 14500])
ax_pd.legend(loc='center right', bbox_to_anchor=(-0.04, 0.5))
ax_pd.set_xlabel('z')
box_pd = ax_pd.get_position()
ax_pd.set_position([box_pd.x0+0.02, box_pd.y0, box_pd.width*0.9, box_pd.height])

ax_t1 = fig1.add_subplot(222)
# TR=20 ms
ax_t1.errorbar(range(0, nz_tr20), t1_noB1_mean_std_tr20[:, 0], t1_noB1_mean_std_tr20[:, 1], label='T1 without B1 correction (TR=20ms)', color='b')
ax_t1.errorbar(range(0, nz_tr20), t1_B1_mean_std_tr20[:, 0], t1_B1_mean_std_tr20[:, 1], label='T1 with B1 correction (TR=20ms)', color='g')
ax_t1.errorbar(range(0, nz_tr20), t1_noB1_mean_std_tr20m[:, 0], t1_noB1_mean_std_tr20m[:, 1], label='T1 without B1 correction (TR=20ms)-matlab', color='b', lw=3.0)  # matlab
ax_t1.errorbar(range(0, nz_tr20), t1_B1_mean_std_tr20m[:, 0], t1_B1_mean_std_tr20m[:, 1], label='T1 with B1 correction (TR=20ms)-matlab', color='g', lw=3.0)  # matlab
# TR=10 ms
ax_t1.errorbar(range(0, nz_tr10), t1_noB1_mean_std_tr10[:, 0], t1_noB1_mean_std_tr10[:, 1], label='T1 without B1 correction (TR=10ms)', color='b', ls='--')
ax_t1.errorbar(range(0, nz_tr10), t1_B1_mean_std_tr10[:, 0], t1_B1_mean_std_tr10[:, 1], label='T1 with B1 correction (TR=10ms)', color='g', ls='--')

# ax_t1.set_ylim([6.28, 6.29])
ax_t1.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
ax_t1.set_xlabel('z')
box_t1 = ax_t1.get_position()
ax_t1.set_position([box_t1.x0-0.05, box_t1.y0, box_t1.width, box_t1.height])

ax_spgr = fig1.add_subplot(223)
# TR=20 ms
ax_spgr.errorbar(range(0, nz_tr20), spgr5_mean_std_tr20[:, 0], spgr5_mean_std_tr20[:, 1], label='SPGR with flip angle = 5 deg (TR=20ms)', color='b')
ax_spgr.errorbar(range(0, nz_tr20), spgr10_mean_std_tr20[:, 0], spgr10_mean_std_tr20[:, 1], label='SPGR with flip angle = 10 deg (TR=20ms)', color='g')
ax_spgr.errorbar(range(0, nz_tr20), spgr20_mean_std_tr20[:, 0], spgr20_mean_std_tr20[:, 1], label='SPGR with flip angle = 20 deg (TR=20ms)', color='r')
ax_spgr.errorbar(range(0, nz_tr20), spgr30_mean_std_tr20[:, 0], spgr30_mean_std_tr20[:, 1], label='SPGR with flip angle = 30 deg (TR=20ms)', color='c')
# TR=10 ms
ax_spgr.errorbar(range(0, nz_tr10), spgr5_mean_std_tr10[:, 0], spgr5_mean_std_tr10[:, 1], label='SPGR with flip angle = 5 deg (TR=10ms)', color='b', ls='--')
ax_spgr.errorbar(range(0, nz_tr10), spgr10_mean_std_tr10[:, 0], spgr10_mean_std_tr10[:, 1], label='SPGR with flip angle = 10 deg (TR=10ms)', color='g', ls='--')
ax_spgr.errorbar(range(0, nz_tr10), spgr20_mean_std_tr10[:, 0], spgr20_mean_std_tr10[:, 1], label='SPGR with flip angle = 20 deg (TR=10ms)', color='r', ls='--')
ax_spgr.errorbar(range(0, nz_tr10), spgr30_mean_std_tr10[:, 0], spgr30_mean_std_tr10[:, 1], label='SPGR with flip angle = 30 deg (TR=10ms)', color='c', ls='--')

ax_spgr.legend(loc='center right', bbox_to_anchor=(-0.04, 0.5))
ax_spgr.set_xlabel('z')

box_spgr = ax_spgr.get_position()
ax_spgr.set_position([box_spgr.x0+0.02, box_spgr.y0, box_spgr.width*0.9, box_spgr.height])


ax_b1 = fig1.add_subplot(224)
ax_b1.errorbar(range(0, nz_tr20), b1_no_smooth_mean_std[:, 0], b1_no_smooth_mean_std[:, 1], label='B1 angle scaling map without smoothing')
ax_b1.errorbar(range(0, nz_tr20), b1_smooth_mean_std[:, 0], b1_smooth_mean_std[:, 1], label='B1 angle scaling map with smoothing')
ax_b1.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
ax_b1.set_xlabel('z')
box_b1 = ax_b1.get_position()
ax_b1.set_position([box_b1.x0-0.05, box_b1.y0, box_b1.width, box_b1.height])


plt.savefig('plot.pdf')

plt.show(block=False)

