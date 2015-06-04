function sct_dmri_moco(varargin)
% process diffusion mri data
%
% USAGE:
% >> sct_dmri_moco() --> calls a GUI
% >> sct_dmri_moco(___,Name,Value)
%
% EXAMPLES:
% >> sct_dmri_moco()
% >> sct_dmri_moco('data','qspace.nii.gz','bvec','bvecs.txt','crop','none','eddy',0)
%
% Option Names:
%     bval  (bval text file)
%     bvec (bvec text file)
%     method : 'b0','dwi'*,'dwi_lowbvalue'
%     crop : 'manual', 'box', 'none'*, 'centerline', 'autobox'
%     eddy : 0 | 1*
%     interp : 'nearestneighbour', 'spline'*, 'sinc'
%     gaussian_mask : <sigma>. Default: 0. Weigth with gaussian mask? Sigma in mm --> std of the kernel. Can be a vector ([sigma_x sigma_y])
%     smooth_moco :   0 | 1*    


dbstop if error
p = inputParser;
crops = {'manual', 'box', 'none', 'centerline', 'autobox'};
addOptional(p,'crop','none',@(x) any(validatestring(x,crops)));
addOptional(p,'eddy',1,@isnumeric);
moco_methods={'b0','dwi','dwi_lowbvalue','none'};
addOptional(p,'method','dwi_lowbvalue',@(x) any(validatestring(x,moco_methods)));
addOptional(p,'crop_margin',25,@isnumeric);
addOptional(p,'data','');
addOptional(p,'bvec','');
addOptional(p,'bval','');
addOptional(p,'smooth_moco',1,@isnumeric);
addOptional(p,'apply_moco_on_croped',1,@isnumeric);
interp={'nearestneighbour', 'spline', 'sinc'};
addOptional(p,'interp','spline',@(x) any(validatestring(x,interp)));
addOptional(p,'gaussian_mask',0,@isnumeric);
addOptional(p,'slicewise',1,@isnumeric);
addOptional(p,'ref',1,@isnumeric);

parse(p,varargin{:})
in=p.Results;
if isempty(in.data)
    [sct.dmri.file,sct.dmri.path] = uigetfile('*.nii;*.nii.gz','Select 4D diffusion data') ;
    if sct.dmri.file==0, return; end
else
    [sct.dmri.path, sct.dmri.file, ext] = fileparts(in.data); if ~isempty(sct.dmri.path), sct.dmri.path=[sct.dmri.path, filesep]; end; sct.dmri.file=[sct.dmri.file,ext];
end
if isempty(in.bvec)
    [sct.dmri.file_bvecs,sct.dmri.path_bvecs] = uigetfile('*','Select bvecs file') ;
    if sct.dmri.file_bvecs==0, return; end
else
    [sct.dmri.path_bvecs, sct.dmri.file_bvecs, ext] = fileparts(in.bvec);  if ~isempty(sct.dmri.path_bvecs), sct.dmri.path_bvecs=[sct.dmri.path_bvecs, filesep]; end; sct.dmri.file_bvecs=[sct.dmri.file_bvecs,ext];
end
if isempty(in.bval)
    [sct.dmri.file_bvals,sct.dmri.path_bvecs] = uigetfile('*','Select bval file') ;
    if sct.dmri.file_bvecs==0, return; end
    sct.dmri.file_bvals=[sct.dmri.path_bvecs filesep sct.dmri.file_bvals];
else
    sct.dmri.file_bvals = in.bval;
end


% Add scripts to path
batch_path= mfilename('fullpath');
spinalcordtoolboxpath=[fileparts(fileparts(fileparts(batch_path))),filesep];
list=dir([spinalcordtoolboxpath, 'src']);
folders_scripts={list(cell2mat({list.isdir})).name};
addpath([spinalcordtoolboxpath,'flirtsch'])
for i_dir=1:length(folders_scripts)
    addpath([spinalcordtoolboxpath, 'src/', folders_scripts{i_dir}]);
end

% =========================================================================
% General Parameters 
% =========================================================================

sct.output_path         = [sct.dmri.path 'dwi_process_data/'];
% TODO: if ~exist('./dwi_process_data','dir')
if exist(sct.output_path)
    unix(['rm -rf ' sct.output_path]);
end
mkdir(sct.output_path);

% Misc
sct.struct.file                     = 'sct'; % structure file name
sct.shell                           = ''; % SHELL run by Matlab. Values are: 'bash' or 'tsh'.
sct.info                            = 'NeuroPoly'; % info
sct.log                             = 'log_process.txt';
sct.outputtype                      = 'NIFTI';





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
sct.dmri.eddy_correct.do			= in.eddy;
sct.dmri.eddy_correct.swapXY		= 0; % Swap X-Y dimension (to have X as phase-encoding direction). If acquisition was axial: set to 1, if sagittal: set to 0.
sct.dmri.eddy_correct.mask_brain	= 0; % Create mask automatically using BET and use the mask to register pairs of opposite directions.   
sct.dmri.eddy_correct.slicewise		= in.slicewise; % binary. Estimate transformation for each slice independently. If you assume eddy-current are not dependent of the Z direction, then put to 0, otherwise 1. Default=1.
sct.dmri.eddy_correct.dof			= which('schedule_TxTy_2mm.sch'); % 'TxSx' | 'TxSxKx'*    Degree of freedom for coregistration of gradient inversed polarity. Tx = Translation along X, Sx = scaling along X, Kx = shearing along X. N.B. data will be temporarily X-Y swapped because FLIRT can only compute shearing parameter along X, not Y
sct.dmri.eddy_correct.fit_transfo	= 0; % Fit transformation parameter (linear least square). Assumes linearity between transformation coefficient (Ty, Sy, Ky) and diffusion gradient amplitude (G). Default=0
sct.dmri.eddy_correct.apply_jacobian= 0; % Apply Jacobian to correct for intensity modulation due to stretching/expansion. Default=1. 
sct.dmri.eddy_correct.interpolation = in.interp; % 'nearestneighbour' | 'spline'* | 'sinc'.
sct.dmri.eddy_correct.outputsuffix	= '_eddy';
sct.dmri.eddy_correct.display_fig	= 0; % 0 | 1*. Display figure of fitted parameters (only if fit_transfo=1). Turn it to 0 if running Matlab without JAVA.
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
sct.dmri.crop.method				= in.crop; % 'manual', 'box', 'none', 'centerline', 'autobox' --> NOTE: if there is no anatomic data proidedprovided, the center_line optino pops a box to select the center line on the DWI data
sct.dmri.crop.file_crop				= 'mask_crop'; % ONLY USED WITH 'manual' METHOD. File name of the mask used for cropping. Put the mask in the first folder (in case of multiple averaging). N.B. The cropping is done on a slice-by-slice basis, i.e., it is possible to define a "non-parallelipipedic" shape.
sct.dmri.crop.size					= '53 31 54 18 0 7'; % ONLY USED WITH 'box' METHOD. Enter numbers as to be used by fslroi. Example: '30 45 25 45 0 16'. 
sct.dmri.crop.margin                = in.crop_margin ; % ONLY USED WITH 'centerline' and 'auto box' METHOD. default : 15
sct.dmri.crop.apply_moco            = in.apply_moco_on_croped ; % Apply moco on croped data or raw data?
sct.dmri.suffix_crop                = '_crop';

% Upsample
sct.dmri.upsample.do                = 0; % use this if you have a malloc error in Flirt. Flirt may need more voxels to compute

% Intra-run motion correction
sct.dmri.moco_intra.method              = in.method; % 'b0','dwi','dwi_lowbvalue','none' (N.B. 'b0' should only be used with data acquired with interspersed b0. Otherwise, PROVIDING SUFFICIENT SNR, use 'dwi').
sct.dmri.moco_intra.smooth_motion       = in.smooth_moco; % Apply a spline in time to estimated motion correction
sct.dmri.schemefile                     = '';
sct.dmri.moco_intra.gaussian_mask       = in.gaussian_mask; % Default: 0. Weigth with gaussian mask? Sigma in mm --> std of the kernel. Can be a vector ([sigma_x sigma_y])
sct.dmri.moco_intra.dwi_group_size      = 10; % number of images averaged for 'dwi' method.
sct.dmri.moco_intra.program             = 'ANTS';% 'FLIRT' or 'SPM' (slicewise not available with SPM.... put slicewise = 0)
sct.dmri.moco_intra.ref                 = num2str(in.ref); % string. Either 'mean_b0' or 'X', X being the number of b0 to use for reference. E.g., sct.dmri.moco_intra.ref = '1' to register data to the first b=0 volume. !!! This flag is only valid if sct.dmri.moco_intra.method = 'b0'
sct.dmri.moco_intra.slicewise		    = in.slicewise; % slice-by-slice motion correction. Put 0 for volume-based moco, 1 otherwise. 
sct.dmri.moco_intra.cost_function_flirt	= 'normcorr'; % 'mutualinfo' | 'woods' | 'corratio' | 'normcorr' | 'normmi' | 'leastsquares'. Default is 'normcorr'.
sct.dmri.moco_intra.cost_function_spm   = 'nmi'; % JULIEN: add other options
sct.dmri.moco_intra.flirt_options       = ['-interp ' in.interp]; % additional FLIRT options. Example: '-dof 6 -interp sinc'. N.B. If gradient non-linearities, it makes sense to use dof=12, otherwise dof=6.
sct.dmri.moco_intra.correct_bvecs       = 0; % correct b-matrix along with motion correction.
sct.dmri.moco_intra.dof                 = which('schedule_TxTy_2mm.sch'); % 'TxTyTzSxSySzKxKyKz' | 'TxSxKxKy' | 'TxSx' | 'TxSxKx'*    Degree of freedom for coregistration of gradient inversed polarity. Tx = Translation along X, Sx = scaling along X, Kx = shearing along X. N.B. data will be temporarily X-Y swapped because FLIRT can only compute shearing parameter along X, not Y
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
% process dmri
% =========================================================================
sct.dmri = j_dmri_initialization(sct.dmri);

j_disp(sct.log,['\n\n\n=========================================================================================================='])
j_disp(sct.log,['   process diffusion data'])
j_disp(sct.log,['=========================================================================================================='])
j_disp(sct.log,['.. Started: ',datestr(now)])

% process dmri
[sct, status] = sct_process_dmri(sct);

save('workspace')



% save structure
save sct sct


% display time
j_disp(sct.log,['\n.. Ended: ',datestr(now)])
j_disp(sct.log,['=========================================================================================================='])
if status, j_disp(sct.log,['---> FIX THE ERROR TO REMOVE THE TERROR !!!']), end
j_disp(sct.log,['\n'])

