function j_batch_process_dmri()
% Process DW-MRI data
% HCP_061
% Zeynep's data
% 
% Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2012-11-18


clear
dbstop if error

% add path to matlab functions
 addpath '/autofs/cluster/connectome/users/jcohen/matlab/common_script'
 addpath '/autofs/cluster/connectome/users/jcohen/matlab/dmri'
 addpath '/autofs/cluster/connectome/users/jcohen/matlab/mri'

% path to gradient nonlinearity distortion correction toolbox
addpath '/autofs/cluster/connectome/users/jcohen/matlab/gradient_nonlin_unwarp';
%addpath '/Users/julien/matlab/toolbox/gradient_nonlin_unwarp';



dmri.input_files				= 'nifti'; % 'dicom' | 'nifti'

% add fields in dmri structure (mainly file names...)
dmri = j_dmri_initialization_v4(dmri);

% bval file general
% file_bvals_general = 'bvals5k_80dir';

% ENTER BVECS FILE TO USE (FSL FORMAT)
fname_bvecs = '/cluster/connectome/users/zsaygin/jQBI120_revGrad';



% =========================================================================
% DICOM Parameters (don't touch if you only have NIFTI)
% =========================================================================
convert_dicom					= 0; % Convert from DICOM to NIFTI. Only need to do it once
get_gradientDir					= 0; % Get gradient directions from DICOM header. Only need to do it once
dmri.dicom.path					= '/cluster/archive/298/siemens/ConnectomA-45074-20120314-164404-796000/';
dmri.dicom.file					= {...
	'187000-000024-*.dcm',...
	};
dmri.input_type					= ''; % If conversion doesn't work, put '-it dicom'. Otherwise, put ''.
dmri.convert2float				= ''; % If you want float precision (notably for diffusion scans), put '-odt float'. Otherwise put ''.
dmri.outputtype					= 'NIFTI'; % NIFTI | NIFTI_GZ



% =========================================================================
% NIFTI Parameters
% =========================================================================
dmri.nifti.file_data_raw		= 'b5k'; % DON'T PUT THE EXTENSION!!!! It can handle nii and nii.gz
dmri.nifti.file_bvecs_raw		= 'bvecs'; % will copy "fname_bvecs" to this file locally
dmri.nifti.file_bvals_raw		= ''; % should be in the same folder as the data. If you don't have bvals file (e.g. for DSI data), then leave it empty.
dmri.nifti.path					= './'; % If everything is local, put './'
dmri.nifti.folder				= {''}; % Default=''



% =========================================================================
% OTHER PARAMETERS
% =========================================================================

dmri.struct.path				= [pwd,filesep]; % current path
dmri.outputtype_final			= 'NIFTI'; % NIFTI | NIFTI_GZ

% gradient non-linearity distortion correction
dmri.grad_nonlin.do				= 1;
dmri.grad_nonlin.gradient_name	= 'coeff_AS302.grad';
dmri.grad_nonlin.FLAG__surface	= 0;
dmri.grad_nonlin.method			= 'direct';
dmri.grad_nonlin.polarity		= 'UNDIS';
dmri.grad_nonlin.biascor		= '1';
dmri.grad_nonlin.interp			= 'cubic';
dmri.grad_nonlin.JacDet			= '0';

% eddy-current correction using the inverse polarity method. N.B. Opposite gradient directions should be acquired next to each other (to minimize the effect of subject motion).
dmri.eddy_correct.do			= 1;
dmri.eddy_correct.swapXY		= 1; % Swap X-Y dimension (to have X as phase-encoding direction). If acquisition was axial: set to 1, if sagittal: set to 0.
dmri.eddy_correct.mask_brain	= 0; % Create mask automatically using BET and use the mask to register pairs of opposite directions.   
dmri.eddy_correct.slicewise		= 0; % binary. Estimate transformation for each slice independently. If you assume eddy-current are not dependent of the Z direction, then put to 0, otherwise 1. Default=1.
dmri.eddy_correct.dof			= 'TxSxKxKy'; % 'TxSx' | 'TxSxKx'*    Degree of freedom for coregistration of gradient inversed polarity. Tx = Translation along X, Sx = scaling along X, Kx = shearing along X. N.B. data will be temporarily X-Y swapped because FLIRT can only compute shearing parameter along X, not Y
dmri.eddy_correct.fit_transfo	= 0; % Fit transformation parameter (linear least square). Assumes linearity between transformation coefficient (Ty, Sy, Ky) and diffusion gradient amplitude (G). Default=0
dmri.eddy_correct.apply_jacobian= 1; % Apply Jacobian to correct for intensity modulation due to stretching/expansion. Default=1. 
dmri.eddy_correct.interpolation = 'trilinear'; % 'nearestneighbour' | 'trilinear'* | 'sinc'.
dmri.eddy_correct.outputsuffix	= '_eddy';
dmri.eddy_correct.display_fig	= 1; % 0 | 1*. Display figure of fitted parameters (only if fit_transfo=1). Turn it to 0 if running Matlab without JAVA.
% dmri.eddy_correct.eddy_first	= 1; % if pairs of inversed-polarity diffusion gradients ARE NOT adjacent in time, it is suggested to apply moco first (hence put the flag to 0). In that case, there will be TWO interpolations. TODO: Need to fix it

% reorientation
dmri.reorient.do				= 0;
dmri.nifti.reorient				= 'LR PA IS'; % if axial acquisition, use LR PA IS
											  % if sagittal, use RL PA IS
dmri.gradients.referential		= 'PRS'; % 'PRS': patient referential -> no correction, 'XYZ': scanner referential -> re-orientation of diffusion vectors
dmri.gradients.flip				= [1 2 3]; % flip gradients along x, y or z. Put [1 2 3] for no flip. Examples: [-2 1 3] (axial @Bay8) ; [1 -2 3] (RL PA IS sagittal @Bay4, to use TrackVis afterwards, flip Y).

% Cropping
dmri.crop.method				= 'none'; % 'manual', 'box', 'none'.
dmri.crop.file_crop				= 'mask_crop'; % ONLY USED WITH 'manual' METHOD. File name of the mask used for cropping. Put the mask in the first folder (in case of multiple averaging). N.B. The cropping is done on a slice-by-slice basis, i.e., it is possible to define a "non-parallelipipedic" shape.
dmri.crop.size					= '40 80 45 80 0 6'; % ONLY USED WITH 'box' METHOD. Enter numbers as to be used by fslroi. Example: '30 45 25 45 0 16'. 

% Intra-session co-registration (co-register the first run of the series into the first run of another series)
dmri.moco_session.do			= 0; % 0,1
dmri.moco_session.fname			= '/cluster/connectome/data/HCP_032/nii/b0_moco_mean.nii'; % file name of the target NIFTI file for intra session co-registration
% dmri.moco_session.cost			= 'normcorr'; % 'mutualinfo', 'woods', 'corratio', 'normcorr', 'normmi', 'leastsquares'

dmri.moco_inter.do				= 0;

% Intra-run motion correction
dmri.moco_intra.method			= 'b0'; % 'b0','dwi','none' (N.B. 'b0' should only be used with data acquired with interspersed b0. Otherwise, PROVIDING SUFFICIENT SNR, use 'dwi').
dmri.moco_intra.ref				= '1'; % string. Either 'mean_b0' or 'X', X being the number of b0 to use for reference. E.g., dmri.moco_intra.ref = '1' to register data to the first b=0 volume. !!! This flag is only valid if dmri.moco_intra.method = 'b0'
dmri.moco_intra.slicewise		= 0; % slice-by-slice motion correction. Put 0 for volume-based moco, 1 otherwise.
dmri.moco_intra.cost			= 'normcorr'; % 'mutualinfo' | 'woods' | 'corratio' | 'normcorr' | 'normmi' | 'leastsquares'. Default is 'normcorr'.
dmri.moco_intra.flirt_options	= '-forcescaling -dof 6 -interp trilinear'; % additional FLIRT options. Example: '-dof 6 -interp sinc'. N.B. If gradient non-linearities, it makes sense to use dof=12, otherwise dof=6.
dmri.moco_intra.correct_bvecs	= 0; % correct b-matrix along with motion correction.
% Experimental stuff
	dmri.moco_intra.improved_b0_moco= 0; % ONLY FOR DWI METHOD: re-run moco on the b=0 by first estimating the transformation matrix (M1) between the mean b=0 image and the mean DWI, then estimating the transformation matrix (M2) between the individual b=0 images and the mean b=0, then applying transformation matrices M1 and M2 to the individual b=0 images.
	dmri.moco_intra.second_passage	= 0; % ONLY FOR DWI METHOD: re-run motion correction after estimating another target based on the 1st motion correction. My experience is that it doesn't bring much improvement...

% Clean DWI dataset
dmri.removeInterspersed_b0		= 1; % remove interspersed b=0 images (e.g. useful for DTK). Default=0.

% Reorder data
dmri.reorder_data.do 			= 0;
dmri.reorder_data.fname_target		= '/cluster/connectome/data/qvecs_515';

% Masking
dmri.mask.method				= 'bet'; % 'manual': Draw a mask using fslview and save it in the specified directory under the name: "nodif_brain_mask.nii.gz". The program will pause to let you draw the mask under fslview, and once done you can just press a key under Matlab to continue the process.
										  % 'auto': generate a mask using a thresholded version of the mean DW image
									      % 'bet': FSL tool. Reference image is the mean dwi
										  % 'copy': Copy an existing mask to the current folder. Indicate the path+file name of the mask to use in the flag fname_mask
										  % 'none': no masking
dmri.mask.ref					= 'b0'; % 'b0' | 'dwi'.
dmri.mask.display				= 0; % display mask and re-generate it via an interative process, asking the user to re-adjust the parameters
dmri.mask.useSameMask			= 1; % if multiple folders, do you want to use the same mask for all folders (1) or create a different mask for each folder (0)?
dmri.mask.bet_threshold			= 0.4; % threshold used by BET (FSL) to generate mask. Smaller values give larger brain outline estimates
dmri.mask.auto.fwhm				= 2; % FWHM for smoothing
dmri.mask.auto.threshold		= 100; % threshold used for intensity-based masking
dmri.mask.manual.ask			= 1; % interrupt the program and wait for the user to create the mask
dmri.mask.copy.fname			= '/cluster/connectome/data/HCP_032/nii/nodif_brain_mask.nii'; % path+file name of the mask to use. ONLY USED WHEN dmri.mask.method='copy' 

% Average folders
dmri.average					= 0; % average multiple folders into 'average/' folder. Put that to 1.

% DTI
dmri.dti.do_each_run			= 0; % compute the tensors for each run (if only one averaging, put that to 1)
dmri.fa.slice					= 5; % slice number to compute the FA
dmri.dti.compute_radial_diffusivity = 0;

% Misc
dmri.struct.file				= 'dmri'; % structure file name
dmri.log						= ['log_process_dmri.txt']; % create a log file of the processing
dmri.shell						= ''; % SHELL run by Matlab. Values are: 'bash' or 'tsh'.
dmri.info						= 'Martinos Center'; % info

% Q-Ball
dmri.dtk.do						= 0; % process qball data
dmri.dtk.file_qball				= 'qball';
dmri.dtk.file_bvecs_dtk			= 'bvecs_dtk';
dmri.dtk.folder_mat				= '/Users/julien/mri/connectome/diffusion_toolkit/';

dmri.delete_raw_data			= 0;


% delete log file
if exist(dmri.log), delete(dmri.log), end

% START THE SCRIPT - DO NOT MODIFY ANYTHING BELOW THIS LINE

j_disp(dmri.log,['\n\n\n=========================================================================================================='])
j_disp(dmri.log,['   Running: batch_process_dmri.m'])
j_disp(dmri.log,['=========================================================================================================='])
j_disp(dmri.log,['.. Started: ',datestr(now)])



% Find which SHELL is running
j_disp(dmri.log,['\nFind which SHELL is running...'])
[status result] = unix('echo $0');
if ~isempty(findstr(result,'bash'))
        dmri.shell = 'bash';
elseif ~isempty(findstr(result,'tsh'))
        dmri.shell = 'tsh';
elseif ~isempty(findstr(result,'tcsh'))
        dmri.shell = 'tsh';
else
        j_disp(dmri.log,['.. Failed to identify shell. Using default.'])
        dmri.shell = 'tsh';
end
j_disp(dmri.log,['.. Running: ',dmri.shell])

% FSL output
if strcmp(dmri.shell,'bash')
        fsloutput = ['export FSLOUTPUTTYPE=',dmri.outputtype,'; ']; % if running BASH
elseif strcmp(dmri.shell,'tsh') || strcmp(dmri.shell,'tcsh')
        fsloutput = ['setenv FSLOUTPUTTYPE ',dmri.outputtype,'; ']; % if you're running C-SHELL
else
        error('Check SHELL field.')
end




% loop over dicom series
if strcmp(dmri.input_files,'dicom')
	nb_dicoms = length(dmri.dicom.file);
%	dmri.nifti.folder = {};
	for i_dicom = 1:nb_dicoms
		j_disp(dmri.log,['\nProcess run ',num2str(i_dicom),'/',num2str(nb_dicoms)])
		j_disp(dmri.log,['-----------------------------------------------'])

		% Get file names
		j_disp(dmri.log,['\nGet file names...'])
		list_fname = dir([dmri.dicom.path,dmri.dicom.file{i_dicom}]);
		nb_files = size(list_fname,1);
		j_disp(dmri.log,['.. Number of files: ',num2str(nb_files)])

		% read dicom to get series number
		j_disp(dmri.log,['\nRead dicom to get series number...'])
		fname = [dmri.dicom.path,list_fname(1).name];
		j_disp(dmri.log,['.. First file name: ',fname])

		% use freesurfer tool to convert to nifti
		if convert_dicom
			j_disp(dmri.log,['\nConvert DICOM to NIFTI using FreeSurfer...'])
			switch(dmri.outputtype)
			case 'NIFTI'
				ext = '.nii';
			case 'NIFTI_GZ'
				ext = '.nii.gz';
			end
			cmd = ['export UNPACK_MGH_DTI=0; mri_convert ',fname,' -o ',[dmri.nifti.path,dmri.nifti.folder{i_dicom},dmri.nifti.file_data_raw,ext,' ',dmri.input_type,' ',dmri.convert2float]];
			j_disp(dmri.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
		end

		% get gradient vectors
		if get_gradientDir
			j_disp(dmri.log,['\nGet gradient vectors...'])
			opt.path_read = [dmri.dicom.path,dmri.dicom.file{i_dicom}];
			opt.path_write = [dmri.nifti.path,dmri.nifti.folder{i_dicom}(1:end-1)];
			opt.file_bvecs = dmri.nifti.file_bvecs_raw;
			opt.file_bvals = dmri.nifti.file_bvals_raw;
			opt.verbose = 1;
			if strcmp(dmri.gradients.referential,'XYZ')
				opt.correct_bmatrix = 1;
			elseif strcmp(dmri.gradients.referential,'PRS')
				opt.correct_bmatrix = 0;
			end
			gradients = j_dmri_gradientsGet(opt);
		end
		
		% copy bvecs/bvals
		copyfile(fname_bvecs,dmri.nifti.file_bvecs_raw)

	end
	
else
	% NIFTI DATA
	nb_dicoms = length(dmri.nifti.folder);

	% copy bvecs/bvals
        copyfile(fname_bvecs,dmri.nifti.file_bvecs_raw)

end


% initialize file names
dmri.nifti.file_data = dmri.nifti.file_data_raw;
dmri.nifti.file_bvecs = dmri.nifti.file_bvecs_raw;
dmri.nifti.file_bvals = dmri.nifti.file_bvals_raw;


% read in gradient vectors
for i_dicom = 1:nb_dicoms

	j_disp(dmri.log,['\nRead in gradient vectors...'])

	% bvecs
	fname_bvecs = [dmri.nifti.path,dmri.nifti.folder{i_dicom},dmri.nifti.file_bvecs];
	j_disp(dmri.log,['.. File bvecs: ',fname_bvecs])
	dmri.gradients.bvecs = textread(fname_bvecs);
	
	% bvals
	if ~isempty(dmri.nifti.file_bvals_raw)
		fname_bvals = [dmri.nifti.path,dmri.nifti.folder{i_dicom},dmri.nifti.file_bvals];
		j_disp(dmri.log,['.. File bvals: ',fname_bvals])
		dmri.gradients.bvals = textread(fname_bvals);
	else
		j_disp(dmri.log,['.. !! bvals file is empty. Must be DSI data.'])
	end
	
	% check directions
	j_disp(dmri.log,['.. Number of directions: ',num2str(size(dmri.gradients.bvecs,1))])
	if ~isempty(dmri.nifti.file_bvals_raw)
		j_disp(dmri.log,['.. Maximum b-value: ',num2str(max(dmri.gradients.bvals)),' s/mm2'])
	end

end


% flip gradient
flip = dmri.gradients.flip;
if flip(1)~=1 | flip(2)~=2 | flip(3)~=3
	j_disp(dmri.log,['\nFlip gradients...'])
	j_disp(dmri.log,['.. flip options: ',num2str(flip)])
	fname_bvecs = [dmri.nifti.path,dmri.nifti.folder{i_dicom},dmri.nifti.file_bvecs];
	gradient = textread(fname_bvecs);
	dmri.nifti.file_bvecs = [dmri.nifti.file_bvecs,'_flip',num2str(flip(1)),num2str(flip(2)),num2str(flip(3))];
	fname_bvecs_new = [dmri.nifti.path,dmri.nifti.folder{i_dicom},dmri.nifti.file_bvecs];
	fid = fopen(fname_bvecs_new,'w');
	for i=1:size(gradient,1)
		G = [sign(flip(1))*gradient(i,abs(flip(1))),sign(flip(2))*gradient(i,abs(flip(2))),sign(flip(3))*gradient(i,abs(flip(3)))];
		fprintf(fid,'%1.10f %1.10f %1.10f\n',G(1),G(2),G(3));
	end
	fclose(fid);
	j_disp(dmri.log,['.. File written: ',fname_bvecs_new])
end	


% save structure
j_disp(dmri.log,['\nSave structure...'])
fname_struct = [dmri.struct.path,dmri.struct.file];
j_disp(dmri.log,['.. Output file: ',fname_struct,'.mat'])
save(fname_struct,'dmri');


% Check if input data is NIFTI or NIFTI_GZ
fname_data = [dmri.nifti.path,dmri.nifti.folder{1},dmri.nifti.file_data];
j_disp(dmri.log,['\nCheck extension of input data: [',fname_data,']...'])
full_name = ls([fname_data,'.*']);
if isempty(strfind(full_name,'.gz'))
	j_disp(dmri.log,['.. NIFTI --> good!'])
else
	j_disp(dmri.log,['.. NIFTI_GZ --> convert to NIFTI'])
	cmd = ['fslchfiletype NIFTI ',fname_data];
	j_disp(dmri.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
end


% process data
[dmri status] = j_dmri_process_data_v11(dmri);


% save structure
save dmri dmri


% display time
j_disp(dmri.log,['\n.. Ended: ',datestr(now)])
j_disp(dmri.log,['=========================================================================================================='])
if status, j_disp(dmri.log,['---> FIX THE ERROR TO REMOVE THE TERROR !!!']), end
j_disp(dmri.log,['\n'])

