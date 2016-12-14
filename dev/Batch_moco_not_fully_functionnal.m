% function j_batch_process()
% Process Multiparametric data
% errsm_03
 
% JULIEN:  NOTES
%	test slice-wise moco for dmri
%	check disco.
%	if we use the box option, make it by default centered, with a width of 31 voxels
% have only ONE log file at the end
% remove intermediate files in the dmri folder

% NOTES FOR YOU:
% DMRI: don't output FLOAT32 data otherwise processing is too long
% estimate DTI parameters and register it to the anat.
% include noise correction on the anat
% checker generation bvals pour mri_convert
% looks like it didn't run 'sct_register_2_anat'
% --> register mtr and DTI metrics to the anat.

% 2013-06-07  17.03


clear all
dbstop if error


% add path to matlab functions
addpath '/Volumes/matlab_shared/common_script'
addpath '/Volumes/matlab_shared/dmri'
addpath '/Volumes/matlab_shared/mri'
addpath ''/Volumes/tanguy/matlab''
addpath '/Volumes/tanguy/matlab/spinalcorddiameter'
addpath '/Volumes/tanguy/matlab/spinalcorddiameter/AddToPath'
addpath '/Volumes/tanguy/matlab/SpinalCordToolbox'
addpath '/Volumes/tanguy/matlab/SpinalCordToolbox/src'
addpath '/Volumes/tanguy/matlab/SpinalCordToolbox/src/dmri'
addpath '/Volumes/tanguy/matlab/SpinalCordToolbox/src/mri'
addpath '/Volumes/tanguy/matlab/SpinalCordToolbox/src/common_script'
addpath '/Volumes/tanguy/matlab/SpinalCordToolbox/src/SpinalCordSegmentation'


%cd('/Network/Servers/django.neuro.polymtl.ca/Volumes/hd2/users_hd2/tanguy/data/Boston/KS_HCP35/delta14')


% =========================================================================
% General Parameters 
% =========================================================================
sct.input_files_type	= 'nifti'; % 'dicom' | 'nifti'
sct.input_path          = '../'; % Put the same as output_path if already processed
sct.output_path         = './';

% Misc
sct.struct.file                     = 'sct'; % structure file name
sct.shell                           = ''; % SHELL run by Matlab. Values are: 'bash' or 'tsh'.
sct.info                            = 'NeuroPoly'; % info
sct.log                             = 'log_process.txt';
sct.outputtype                      = 'NIFTI';





% =========================================================================
% DICOM Parameters (don't touch if you only have NIFTI)
% =========================================================================

sct.dicom.anat.folder             = '14-tse_spc_1mm_p2_FOV384_bottom';
sct.dicom.dmri.folder             = '02-ep2d_diff_2drf_0.8mm_24dir';
sct.dicom.mtr_ON.folder          = '';
sct.dicom.mtr_OFF.folder         = '';
sct.dicom.disco.folder           = ''; % folder to EPI acquired with reversed phase-encoding gradients


sct.convert_dicom.program           ='DCM2NII'; % MRI_CONVERT or DCM2NII
sct.convert_dicom.input_type		= ''; % If conversion doesn't work, put '-it dicom'. Otherwise, put ''.
sct.convert_dicom.convert2float		= ''; % If you want float precision (notably for diffusion scans), put '-odt float'. Otherwise put ''.


% =========================================================================
% NIFTI Parameters (don't touch if you only have DICOM)
% =========================================================================
% File chosen is the first .nii file in each folders
sct.nifti.anat.folder             = '';
sct.nifti.dmri.folder             = 'data_crop_eddy_interp';
sct.nifti.mtr_ON.folder           = '';
sct.nifti.mtr_OFF.folder          = '';
sct.nifti.disco.folder            = '';




% =========================================================================
% Spinal Cord Segmentation Parameters 
% =========================================================================
sct.segmentation.do = 0;
sct.segmentation.image_type = 2;            % 1=T1; 2=T2
sct.segmentation.interval = 10 ;             % Interval in mm between two slices for the initialization

sct.segmentation.nom_radius= 5;            % Nominal radius in mm that reprensents the initial estimate
sct.segmentation.tolerance = 0.01;          % Percentage of the nominal radius that is used as the criterion to determine convergence
sct.segmentation.ratio_criteria = 0.05;     % Percentage of radius that must meet the tolerance factor to increment the coefficients

sct.segmentation.num_angles = 64;           % Number of angles used
sct.segmentation.update_multiplier = 0.8;   % Multiplies the force applied to deform the radius
sct.segmentation.shear_force_multiplier= 0.5;  % Multiplies the shear force used to stay near the user defined center line. 
sct.segmentation.max_coeff_horizontal = 10; % Maximal coefficient used to smooth the radius in the horizontal plane
sct.segmentation.max_coeff_vertical = 10;   % Maximal coefficient used to smooth the radius in the vertical direction (depends on the number of slices)
sct.segmentation.centerline='centerline';
sct.segmentation.surface='surface';
sct.segmentation.log = 'log_segmentation';


% =========================================================================
% DMRI Parameters
% =========================================================================



% gradient non-linearity distortion correction
sct.dmri.grad_nonlin.do				= 0;
sct.dmri.grad_nonlin.gradient_name	= 'coeff_AS302.grad';
sct.dmri.grad_nonlin.FLAG__surface	= 0;
sct.dmri.grad_nonlin.method			= 'direct';
sct.dmri.grad_nonlin.polarity		= 'UNDIS';
sct.dmri.grad_nonlin.biascor		= '1';
sct.dmri.grad_nonlin.interp			= 'cubic';
sct.dmri.grad_nonlin.JacDet			= '0';

% eddy-current correction using the inverse polarity method. N.B. Opposite gradient directions should be acquired next to each other (to minimize the effect of subject motion).
sct.dmri.eddy_correct.do			= 0;
sct.dmri.eddy_correct.swapXY		= 1; % Swap X-Y dimension (to have X as phase-encoding direction). If acquisition was axial: set to 1, if sagittal: set to 0.
sct.dmri.eddy_correct.mask_brain	= 0; % Create mask automatically using BET and use the mask to register pairs of opposite directions.   
sct.dmri.eddy_correct.slicewise		= 1; % binary. Estimate transformation for each slice independently. If you assume eddy-current are not dependent of the Z direction, then put to 0, otherwise 1. Default=1.
sct.dmri.eddy_correct.dof			= 'TxSxKxKy'; % 'TxSx' | 'TxSxKx'*    Degree of freedom for coregistration of gradient inversed polarity. Tx = Translation along X, Sx = scaling along X, Kx = shearing along X. N.B. data will be temporarily X-Y swapped because FLIRT can only compute shearing parameter along X, not Y
sct.dmri.eddy_correct.fit_transfo	= 0; % Fit transformation parameter (linear least square). Assumes linearity between transformation coefficient (Ty, Sy, Ky) and diffusion gradient amplitude (G). Default=0
sct.dmri.eddy_correct.apply_jacobian= 0; % Apply Jacobian to correct for intensity modulation due to stretching/expansion. Default=1. 
sct.dmri.eddy_correct.interpolation = 'trilinear'; % 'nearestneighbour' | 'trilinear'* | 'sinc'.
sct.dmri.eddy_correct.outputsuffix	= '_eddy';
sct.dmri.eddy_correct.display_fig	= 1; % 0 | 1*. Display figure of fitted parameters (only if fit_transfo=1). Turn it to 0 if running Matlab without JAVA.
% sct.dmri.eddy_correct.eddy_first	= 1; % if pairs of inversed-polarity diffusion gradients ARE NOT adjacent in time, it is suggested to apply moco first (hence put the flag to 0). In that case, there will be TWO interpolations. TODO: Need to fix it


% epi distortion correction using reversed gradient technique
sct.disco.do                    = 0;
sct.disco.suffixe_output        = '_disco'; % suffixe for corrected images
sct.disco.switchPlusMinus		= 0; % if distortions look worse after correction, put this parameter to 1.
sct.disco.correct_intensity     = 1; % do intensity correction (Jacobian matrix). Default = 1.
sct.disco.thresh_quantile		= 40; % in percent. Default is 40.
sct.disco.interpolate			= 4; % Default = 4.
sct.disco.smoothing             = 3; % 3-D gaussian smoothing of the deformation field. SHOULD BE AN ODD NUMBER. Default = 3. Put 0 for no smoothing.
sct.disco.interp_method         = 'linear';


% reorientation
sct.dmri.reorient.do				= 0;
sct.dmri.nifti.reorient				= 'LR PA IS'; % if axial acquisition, use LR PA IS
											      % if sagittal, use RL PA IS
sct.dmri.gradients.referential		= 'PRS'; % 'PRS': patient referential -> no correction, 'XYZ': scanner referential -> re-orientation of diffusion vectors
sct.dmri.gradients.flip				= [1 2 3]; % flip gradients along x, y or z. Put [1 2 3] for no flip. Examples: [-2 1 3] (axial @Bay8) ; [1 -2 3] (RL PA IS sagittal @Bay4, to use TrackVis afterwards, flip Y).

% Cropping
sct.dmri.crop.method				= 'none'; % 'manual', 'box', 'none', 'center_line', 'auto box' --> NOTE: if there is no anatomic data proidedprovided, the center_line optino pops a box to select the center line on the DWI data
sct.dmri.crop.file_crop				= 'mask_crop'; % ONLY USED WITH 'manual' METHOD. File name of the mask used for cropping. Put the mask in the first folder (in case of multiple averaging). N.B. The cropping is done on a slice-by-slice basis, i.e., it is possible to define a "non-parallelipipedic" shape.
sct.dmri.crop.size					= '53 31 54 18 0 7'; % ONLY USED WITH 'box' METHOD. Enter numbers as to be used by fslroi. Example: '30 45 25 45 0 16'. 
sct.dmri.crop.margin                = 15 ; % ONLY USED WITH 'centerline' METHOD. default : 15
sct.dmri.suffix_crop                = '_crop';

% Upsample
sct.dmri.upsample.do                = 0; % use this if you have a malloc error in Flirt. Flirt may need more voxels to compute

% Intra-run motion correction
sct.dmri.moco_intra.method              = 'b0'; % 'b0','dwi','none' (N.B. 'b0' should only be used with data acquired with interspersed b0. Otherwise, PROVIDING SUFFICIENT SNR, use 'dwi').
sct.dmri.moco_intra.program             = 'FLIRT';% 'FLIRT' or 'SPM' (slicewise not available with SPM.... put slicewise = 0)
sct.dmri.moco_intra.ref                 = 'b0_mean'; % string. Either 'mean_b0' or 'X', X being the number of b0 to use for reference. E.g., sct.dmri.moco_intra.ref = '1' to register data to the first b=0 volume. !!! This flag is only valid if sct.dmri.moco_intra.method = 'b0'
sct.dmri.moco_intra.slicewise		    = 1; % slice-by-slice motion correction. Put 0 for volume-based moco, 1 otherwise. 
sct.dmri.moco_intra.cost_function_flirt	= 'corratio'; % 'mutualinfo' | 'woods' | 'corratio' | 'normcorr' | 'normmi' | 'leastsquares'. Default is 'normcorr'.
sct.dmri.moco_intra.cost_function_spm   = 'nmi'; % JULIEN: add other options
sct.dmri.moco_intra.flirt_options       = '-forcescaling -interp trilinear'; % additional FLIRT options. Example: '-dof 6 -interp sinc'. N.B. If gradient non-linearities, it makes sense to use dof=12, otherwise dof=6.
sct.dmri.moco_intra.correct_bvecs       = 0; % correct b-matrix along with motion correction.
sct.dmri.moco_intra.dof                 = 'TxTy'; % 'TxTyTzSxSySzKxKyKz' | 'TxSxKxKy' | 'TxSx' | 'TxSxKx'*    Degree of freedom for coregistration of gradient inversed polarity. Tx = Translation along X, Sx = scaling along X, Kx = shearing along X. N.B. data will be temporarily X-Y swapped because FLIRT can only compute shearing parameter along X, not Y
sct.dmri.suffix_moco                    = '_moco';

% Clean DWI dataset
sct.dmri.removeInterspersed_b0		= 0; % remove interspersed b=0 images (e.g. useful for DTK). Default=0.

% Reorder data
sct.dmri.reorder_data.do 			= 0;
sct.dmri.reorder_data.fname_target		= '/cluster/connectome/data/qvecs_515';

% Masking
sct.dmri.mask.method				= 'none'; % 'manual': Draw a mask using fslview and save it in the specified directory under the name: "nodif_brain_mask.nii.gz". The program will pause to let you draw the mask under fslview, and once done you can just press a key under Matlab to continue the process.
										  % 'auto': generate a mask using a thresholded version of the mean DW image
									      % 'bet': FSL tool. Reference image is the mean dwi
										  % 'copy': Copy an existing mask to the current folder. Indicate the path+file name of the mask to use in the flag fname_mask
										  % 'none': no masking
sct.dmri.mask.ref					= 'b0'; % 'b0' | 'dwi'.
sct.dmri.mask.display				= 0; % display mask and re-generate it via an interative process, asking the user to re-adjust the parameters
sct.dmri.mask.bet_threshold			= 0.4; % threshold used by BET (FSL) to generate mask. Smaller values give larger brain outline estimates
sct.dmri.mask.auto.fwhm				= 2; % FWHM for smoothing
sct.dmri.mask.auto.threshold		= 100; % threshold used for intensity-based masking
sct.dmri.mask.manual.ask			= 1; % interrupt the program and wait for the user to create the mask
sct.dmri.mask.copy.fname			= '/cluster/connectome/data/HCP_032/nii/nodif_brain_mask.nii'; % path+file name of the mask to use. ONLY USED WHEN sct.dmri.mask.method='copy' 

% Average folders
sct.dmri.average					= 0; % average multiple folders into 'average/' folder. Put that to 1.

% DTI
sct.dmri.dti.do_each_run			= 0; % compute the tensors for each run (if only one averaging, put that to 1)
sct.dmri.fa.slice					= 5; % slice number to compute the FA
sct.dmri.dti.compute_radial_diffusivity = 0;

% Q-Ball
sct.dmri.dtk.do						= 0; % process qball data
sct.dmri.dtk.file_qball				= 'qball';
sct.dmri.dtk.file_bvecs_dtk			= 'bvecs_dtk';
sct.dmri.dtk.folder_mat				= '/Users/julien/mri/connectome/diffusion_toolkit/';

sct.dmri.delete_raw_data			= 0;


% =========================================================================
% MTR Parameters
% =========================================================================

sct.mtr.fname_output    = 'MTR'; % file name of the resulting MTR
sct.mtr.crop.do         = 1; % crop MTR files for better registration with anat. NEED SEGMENTATION.DO = 1 
sct.mtr.crop.margin     = 20; % crop margin to centerline. default : 15
sct.mtr.cost_spm_coreg  = 'nmi'; % coregistration cost function used by SPM, between MT_ON and MT_OFF. Default : 'nmi' 




% =========================================================================
% ANAT Parameters
% =========================================================================

sct.anat.estimate   = 0; % if 0 registration is based on header files informations
sct.anat.cost_spm_coreg = 'ncc'; % coregistration cost function used by SPM between b=0 and anat, and between MTR and anat. Default : 'ncc'
sct.anat.path            = [sct.output_path, 'anat/'];
sct.anat.log        = 'log_process_sct.anat.txt';

% delete log file
if exist(sct.log), delete(sct.log), end


% =========================================================================
% START THE SCRIPT - DO NOT MODIFY ANYTHING BELOW THIS LINE
% =========================================================================

j_disp(sct.log,['\n\n\n=========================================================================================================='])
j_disp(sct.log,['   Running: batch_process_sct.dmri.m'])
j_disp(sct.log,['=========================================================================================================='])
j_disp(sct.log,['.. Started: ',datestr(now)])


% save structure
j_disp(sct.log,['\nSave structure...'])
fname_struct = [sct.output_path,sct.struct.file];
j_disp(sct.log,['.. Output file: ',fname_struct,'.mat'])
save(fname_struct,'sct');

% =========================================================================
% Conversion dcm2nii
% =========================================================================

[sct, status] = sct_dcm_to_nii_v4(sct);

% =========================================================================
% Spinal Cord Segmentation
% =========================================================================
if sct.segmentation.do
    if isfield(sct.anat,'folder')
        
        j_disp(sct.log,['\n\n\n=========================================================================================================='])
        j_disp(sct.log,['   Running: sct_sct.segmentation.m'])
        j_disp(sct.log,['=========================================================================================================='])
        j_disp(sct.log,['.. Started: ',datestr(now)])
        
        % process segentation
        sct.segmentation.path_name_out = [sct.output_path,'anat/Spinal_Cord_Segmentation/'];
        sct.segmentation.path_name_in  = [sct.output_path,'anat/'];
        sct_segmentation (sct);
    end
end
save('workspace')

% =========================================================================
% process dmri
% =========================================================================
sct.dmri = j_dmri_initialization_v4(sct.dmri);

if isfield(sct.dmri,'folder')
    j_disp(sct.log,['\n\n\n=========================================================================================================='])
    j_disp(sct.log,['   process diffusion data'])
    j_disp(sct.log,['=========================================================================================================='])
    j_disp(sct.log,['.. Started: ',datestr(now)])
    
    % process dmri
    sct.dmri.path = [sct.output_path,'dmri/'];
    [sct, status] = sct_process_dmri(sct);
end

save('workspace')

% =========================================================================
% process mtr
% =========================================================================
   
if isfield(sct.mtr,'folder')
    % process mtr
    sct.mtr.path = [sct.output_path,'mtr/'];
    sct_process_mtr_v1(sct);
end


save('workspace')

% =========================================================================
% process anat
% =========================================================================

if isfield(sct.anat,'folder')
    j_disp(sct.log,['\n\n\n=========================================================================================================='])
    j_disp(sct.log,['   registrate datas with anatonical file'])
    j_disp(sct.log,['=========================================================================================================='])
    j_disp(sct.log,['.. Started: ',datestr(now)])
    
 
    % process anat
    sct_register_2_anat(sct);
end

% save structure
save sct sct


% display time
j_disp(sct.log,['\n.. Ended: ',datestr(now)])
j_disp(sct.log,['=========================================================================================================='])
if status, j_disp(sct.log,['---> FIX THE ERROR TO REMOVE THE TERROR !!!']), end
j_disp(sct.log,['\n'])