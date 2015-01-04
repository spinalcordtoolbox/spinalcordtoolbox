% =========================================================================
% SCRIPT
% j_dmri_initialization.m
%
% update dmri structure
% 
% COMMENTS
% Julien Cohen-Adad 2010-04-09
% =========================================================================
function dmri = j_dmri_initialization(dmri)


dmri.b0							= 50; % in some sequences, the b=0 is not 0 but something like 50. To assign b=50 as b=0, this parameter allows to define the b=0 image as being b<='given threshold'
dmri.nifti.file_data_raw		= 'data_raw';
dmri.nifti.file_data_mean		= 'data_mean.nii';
dmri.nifti.file_data_std			= 'data_std.nii';
dmri.nifti.file_tsnr				= 'tsnr.nii';
dmri.nifti.file_bvecs_raw		= 'bvecs_raw';
dmri.nifti.file_bvals_raw		= 'bvals_raw';
dmri.nifti.file_dti				= 'dti';
dmri.nifti.file_bvecs_moco_intra= 'bvecs';
dmri.nifti.file_bvals_moco_intra= 'bvals';
dmri.nifti.file_data_crop		= 'data_crop';
dmri.nifti.file_data_moco_intra = 'data_moco_concat';
dmri.nifti.file_data_reorient	= 'data';
dmri.nifti.file_data_moco		= 'data_moco';
dmri.nifti.file_data_firstvols	= 'tmp.data_firstvols';
dmri.nifti.file_data_firstvols_mean = 'tmp.data_firstvols_mean';
dmri.nifti.file_datasub_ref		= 'tmp.datasub_ref_';
dmri.nifti.file_nodif			= 'nodif.nii';
dmri.nifti.file_b0_intra		= 'b0';
dmri.nifti.file_b0_intra_moco	= 'b0_moco';
dmri.nifti.file_b0_intra_mean	= 'b0_mean';
dmri.nifti.file_nodif_mean		= 'nodif_mean.nii.gz';
dmri.nifti.file_nodif_mean_moco	= 'nodif_mean_moco.nii.gz';
dmri.nifti.file_moco_intra_mat	= 'moco_intra_mat_';
dmri.nifti.file_mask			= 'nodif_brain_mask';
dmri.nifti.file_dwi				= 'dwi.nii';
dmri.nifti.file_dwi_mean		= 'dwi_mean';
% dmri.nifti.file_dwi_sub			= 'dwi_sub_';
% dmri.nifti.file_dwi_eddy		= 'dwi_eddy_';
dmri.nifti.file_dwi_with_dwi_mean= 'dwi_with_dwi_mean.nii';
dmri.nifti.file_dwi_with_dwi_mean_eddy= 'dwi_with_dwi_mean_eddy.nii';
dmri.nifti.file_dwi_eddy		= 'dwi_eddy.nii';
dmri.nifti.file_data_eddy		= 'data_moco_intra_eddy.nii';
dmri.nifti.file_b0				= 'b0';
dmri.nifti.file_b0_moco			= 'b0_moco';
dmri.moco_inter.file_b0			= 'b0_first'; % file name used to perfor inter-run motion correction
dmri.moco_inter.file_moco		= 'b0_moco'; % file name used to perfor inter-run motion correction
dmri.moco_inter.file_mat		= 'mat_moco_inter';
dmri.moco_inter.file_mat_global	= 'mat_moco_global';
dmri.moco_session.file_mat		= 'mat_moco_session';
dmri.moco_session.file_moco		= 'b0_session';
dmri.nifti.file_datasub			= 'tmp.datasub_';
dmri.nifti.file_datamoco		= 'tmp.datamoco_';
dmri.nifti.folder_average		= 'average_';
dmri.nifti.file_data_final		= 'data';
