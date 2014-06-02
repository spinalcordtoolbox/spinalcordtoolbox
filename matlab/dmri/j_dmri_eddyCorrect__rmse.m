function j_dmri_eddyCorrect__rmse(fname_data, fname_bvecs, opt)
% =========================================================================
% 
% Compute MSE to assess efficiency of distortion correction in DW images. Need data with back-to-back distortions.
% 
% 
% INPUT
% -------------------------------------------------------------------------
% fname_data			string. Data file name. DO NOT PUT THE FILE EXTENSION!!
% fname_bvecs			string. Bvecs file name.
% (opt)
%	folder_mat			string		folder where to put mat files. Default='mat_eddy/'
%	split_data			0 | 1*   If data are already splitted, then indicated the file name in the following flag.
%	  fname_data_splitT	string		Default='tmp.data_splitT'
%   swapXY				0*| 1   Swap X-Y dimension (to have X as phase-encoding direction). If acquisition was axial: set to 1, if sagittal: set to 0.
%   mask_brain			0 | 1*  Create mask automatically using BET and use the mask to register pairs of opposite directions.   
%   min_norm			float.	Minimim norm of two opposite gradient directions (it should be 0 if they are in perfect opposition, but for some reasons, e.g., truncated values, it is note the case. Suggested value=0.001).
%	dof					'TxSx' | 'TxSxKx' | 'TxSxKxKy'   Degree of freedom for coregistration of gradient inversed polarity. Tx = Translation along X, Sx = scaling along X, Kx = shearing along X. N.B. data will be temporarily X-Y swapped because FLIRT can only compute shearing parameter along X, not Y
%   interpolation		'nearestneighbour' | 'trilinear' | 'sinc'*.   Final interpolation
%   cost_function		mutualinfo | corratio | normcorr* | normmi | leastsq | labeldiff
%   outputtype			NIFTI* | NIFTI_GZ
%   outputsuffix		string. Default = '_eddy'
%	fname_log			string
%   slicewise			0 | 1*   Estimate transformation for each slice independently. If you assume eddy-current are not dependent of the Z direction, then put to 0, otherwise 1. Default=1.
%   find_transfo		0 | 1*   Find transformation.
%   fit_transfo			0*| 1    Fit transformation parameter (linear least square)
%	  display_fig		0 | 1*   Display figure of fitted parameters (only if fit_transfo=1). Turn it to 0 if running Matlab without JAVA.
%   apply_transfo		0 | 1*   Estimate and apply transfo (=1) or only estimate (=0).
%   apply_jacobian		0 | 1*	 Apply Jacobian to correct for intensity modulation due to stretching/expansion
%   merge_back			0 | 1*   Merge data back. 
%   compute_rmse		0 | 1*   Compute RMSE for assessing efficiency of distortion correction
%     rmse_norm_range	2x float range of norm of qvecs at which RMSE should be computed. Default = [0.99 1.01] (to account for imprecisions). For DSI_515, use [0.59 0.61].
%     rmse_smooth		0 | 1*	 Smooth to compute RMSE (to get rid of variations induced by noise).	
% -------------------------------------------------------------------------
% 
% OUTPUT
% -------------------------------------------------------------------------
% -
% -------------------------------------------------------------------------
%
% Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2012-02-07: Created
% =========================================================================


% debug if error
dbstop if error

% Check number of arguments
if nargin < 2
	help j_dmri_eddyCorrect__rmse
	return
end


% INITIALIZATION
if ~exist('opt'), opt = []; end
% if isfield(opt,'dof'), dof = opt.dof; else dof = 'TxSxKx'; end
% if isfield(opt,'interpolation'), interpolation = opt.interpolation; else interpolation = 'sinc'; end
% if isfield(opt,'cost_function'), cost_function = opt.cost_function; else cost_function = 'normcorr'; end
% if isfield(opt,'min_norm'), min_norm = opt.min_norm; else min_norm = 0.001; end
% if isfield(opt,'outputtype'), outputtype = opt.outputtype; else outputtype = 'NIFTI'; end
% if isfield(opt,'outputsuffix'), outputsuffix = opt.outputsuffix; else outputsuffix = '_eddy'; end
if isfield(opt,'fname_log'), fname_log = opt.fname_log; else fname_log = 'log_j_dmri_eddyCorrect.txt'; end
% if isfield(opt,'swapXY'), swapXY = opt.swapXY; else swapXY = 0; end
% if isfield(opt,'mask_brain'), mask_brain = opt.mask_brain; else mask_brain = 1; end
% if isfield(opt,'split_data'), split_data = opt.split_data; else split_data = 1; end
% if isfield(opt,'fname_data_splitT'), fname_data_splitT = opt.fname_data_splitT; else fname_data_splitT = 'tmp_moco.data_splitT'; end
% % if isfield(opt,'fname_mat'), fname_mat = opt.fname_mat; else fname_mat = ''; end
% if isfield(opt,'folder_mat'), folder_mat = opt.folder_mat; else folder_mat = 'mat_eddy/'; end
% % if isfield(opt,'shell'), shell = opt.shell; else shell = ''; end
% if isfield(opt,'apply_transfo'), apply_transfo = opt.apply_transfo; else apply_transfo = 1; end
% if isfield(opt,'apply_jacobian'), apply_jacobian = opt.apply_jacobian; else apply_jacobian = 1; end
% if isfield(opt,'slicewise'), slicewise = opt.slicewise; else slicewise = 1; end
% if isfield(opt,'find_transfo'), find_transfo = opt.find_transfo; else find_transfo = 1; end
% if isfield(opt,'fit_transfo'), fit_transfo = opt.fit_transfo; else fit_transfo = 0; end
% if isfield(opt,'merge_back'), merge_back = opt.merge_back; else merge_back = 1; end
% if isfield(opt,'compute_rmse'), compute_rmse = opt.compute_rmse; else compute_rmse = 1; end
if isfield(opt,'rmse_norm_range'), rmse_norm_range = opt.rmse_norm_range; else rmse_norm_range = [0.99 1.01]; end
if isfield(opt,'rmse_smooth'), rmse_smooth = opt.rmse_smooth; else rmse_smooth = 2; end
if isfield(opt,'display_fig'), display_fig = opt.display_fig; else display_fig = 1; end
if isfield(opt,'min_norm'), min_norm = opt.min_norm; else min_norm = 0.001; end
if isfield(opt,'mask_brain'), mask_brain = opt.mask_brain; else mask_brain = 1; end


folder_eddy = ['tmp_eddy',filesep];


% =========================================================================
% START THE SCRIPT
% =========================================================================

% START FUNCTION
j_disp(fname_log,['\n\n\n=========================================================================================================='])
j_disp(fname_log,['   Running: j_dmri_eddyCorrect__rmse.m'])
j_disp(fname_log,['=========================================================================================================='])
j_disp(fname_log,['.. Started: ',datestr(now)])


% Check parameters
j_disp(fname_log,['\nCheck parameters:'])
j_disp(fname_log,['.. Input data:        ',fname_data])
j_disp(fname_log,['.. bvecs file:        ',fname_bvecs])
% j_disp(fname_log,['.. swapXY:            ',num2str(swapXY)])
% j_disp(fname_log,['.. mask_brain:        ',num2str(mask_brain)])
% j_disp(fname_log,['.. split_data:        ',num2str(split_data)])
% j_disp(fname_log,['.. fname_data_splitT: ',fname_data_splitT])
% j_disp(fname_log,['.. Degree of freedom: ',dof])






folder_eddy = ['tmp_eddy',filesep];
if ~exist(folder_eddy), mkdir(folder_eddy), end


% Find which SHELL is running
j_disp(fname_log,['\nFind which SHELL is running...'])
[status result] = unix('echo $0');
if ~isempty(findstr(result,'bash'))
	shell = 'bash';
elseif ~isempty(findstr(result,'tsh'))
	shell = 'tsh';
elseif ~isempty(findstr(result,'tcsh'))
	shell = 'tcsh';
else
	j_disp(fname_log,['.. Failed to identify shell. Using default.'])
	shell = 'bash';
end
j_disp(fname_log,['.. Running: ',shell])



% FSL output
if strcmp(shell,'bash')
	fsloutput = ['export FSLOUTPUTTYPE=NIFTI; '];
elseif strcmp(shell,'tsh') || strcmp(shell,'tcsh') 
	fsloutput = ['setenv FSLOUTPUTTYPE NIFTI; '];
end




% get data dimensions
j_disp(fname_log,['\nGet dimensions of the data...'])
cmd = ['fslsize ',fname_data]; [status result] = unix(cmd); if status, error(result); end
dims = j_mri_getDimensions(result);
nx = dims(1);
ny = dims(2);
nz = dims(3);
nt = dims(4);
j_disp(fname_log,['.. ',num2str(nx),' x ',num2str(ny),' x ',num2str(nz),' x ',num2str(nt)])
clear img



% split into T dimension
folder_dataSplit = folder_eddy;
j_disp(fname_log,['\nSplit along T dimension...'])
cmd = [fsloutput,'fslsplit ',fname_data,' ',folder_dataSplit,'tmp.data_splitT'];
j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
numT = j_numbering(nt,4,0);



% Identify pairs of opposite gradient directions
j_disp(fname_log,['\nIdentify pairs of opposite gradient directions...'])
bvecs = textread(fname_bvecs);
iN = 1;
opposite_gradients = {};
index_b0 = []; % b0 index for masking
for iT=1:nt
	for jT=1:nt
		if norm(bvecs(iT,:)+bvecs(jT,:))<min_norm && norm(bvecs(iT,:))~=0 && iT<jT
			j_disp(fname_log,['.. Opposite gradient for #',num2str(iT),' is: #',num2str(jT)])
			opposite_gradients{iN} = [iT,jT];
			iN = iN+1;
			break
		end
	end
	if norm(bvecs(iT,:)) == 0
		index_b0 = [index_b0 iT];
	end
end
nb_oppositeGradients = length(opposite_gradients);
j_disp(fname_log,['.. Number of gradient directions: ',num2str(2*nb_oppositeGradients), ' (2*',num2str(nb_oppositeGradients),')'])
j_disp(fname_log,['.. Index b=0: ',num2str(index_b0)])



% Create mask of the brain using b=0 images
if mask_brain
	j_disp(fname_log,['\nCreate mask of the brain using b=0 images...'])
	% merge b=0
	nb_b0 = length(index_b0);
	fname_b0_merge = [folder_eddy,'b0_merge'];
	cmd = [fsloutput,'fslmerge -t ',fname_b0_merge];
	for iT = 1:nb_b0
		cmd = [cmd,' ',folder_dataSplit,'tmp.data_splitT',numT{index_b0(iT)}];
	end
	j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	% Average b=0 images
	fname_b0_mean = [folder_eddy,'b0_mean'];
	cmd = [fsloutput,'fslmaths ',fname_b0_merge,' -Tmean ',fname_b0_mean];
	j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	% create mask using BET
	cmd = [fsloutput,'bet2 ',fname_b0_mean,' brain_eddy -m -n -f 0.3'];
	j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	fname_mask = 'brain_eddy_mask';
end





% ================================================================================================================================
%	RMSE
% ================================================================================================================================
% N.B. Do that in Matlab as opposed to via FSL command, because calculating
% the Squared error and then save it to NIFTI can create overflow error
% because of the signed format of the NIFTI data (i.e., very high value
% would become negative).


j_disp(fname_log,['\n\n\nCompute RMSE'])
j_disp(fname_log,['------------------------------------------------------------------------------------'])

% Identify directions to use for RMSE
j_disp(fname_log,['\nIdentify directions to use for RMSE...'])
norm_bvecs = [];
rmse_oppositeGradients = [];
for iN=1:nb_oppositeGradients
	norm_bvecs(iN) = norm(bvecs(opposite_gradients{iN}(1),:));
	if norm_bvecs(iN) >= rmse_norm_range(1) && norm_bvecs(iN) <= rmse_norm_range(2)
		rmse_oppositeGradients = cat(1,rmse_oppositeGradients,iN);
	end
end
nb_oppositeGradients_rmse = length(rmse_oppositeGradients);
j_disp(fname_log,['.. Range of norm: [',num2str(rmse_norm_range(1)),',',num2str(rmse_norm_range(2)),']'])
j_disp(fname_log,['.. Number of directions inside that range: ',num2str(nb_oppositeGradients_rmse)])


% RMSE map of uncorrected data
% -------------------------------------------------------------
j_disp(fname_log,['\nCompute normalized MSE of uncorrected data...'])

% loop across direction pairs
se = zeros(nb_oppositeGradients_rmse,nx,ny,nz);
for iN=1:nb_oppositeGradients_rmse
	j_disp(fname_log,['.. Pair ',num2str(iN),'/',num2str(nb_oppositeGradients_rmse)])

	% plus
	i_plus = opposite_gradients{rmse_oppositeGradients(iN)}(1);
	fname_plus = [folder_dataSplit,'tmp.data_splitT',numT{i_plus}];
	if rmse_smooth
		cmd = [fsloutput,'fslmaths ',fname_plus,' -s ',num2str(rmse_smooth),' ',fname_plus,'_smooth'];
		j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
		fname_plus = [fname_plus,'_smooth'];
	end
	[data_plus,dims,scales,bpp,endian] = read_avw(fname_plus);
	% minus
	i_minus = opposite_gradients{rmse_oppositeGradients(iN)}(2);
	fname_minus = [folder_dataSplit,'tmp.data_splitT',numT{i_minus}];
	if rmse_smooth
		cmd = [fsloutput,'fslmaths ',fname_minus,' -s ',num2str(rmse_smooth),' ',fname_minus,'_smooth'];
		j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
		fname_minus = [fname_minus,'_smooth'];
	end
	[data_minus,dims,scales,bpp,endian] = read_avw(fname_minus);
	% compute squared error
	se(1,:,:,:) = (data_plus - data_minus).^2;
end
% compute normalized RMSE
norm_rmse = squeeze(mean(se,1))/nb_oppositeGradients_rmse;
% save volume
fname_rmse = ['MSE_s',num2str(rmse_smooth),'_',fname_data];
j_disp(fname_log,['.. output name: ',fname_rmse])
save_avw(norm_rmse,fname_rmse,'f',scales(1:3));
j_disp(fname_log,['\nCopy geometric information from ',[folder_dataSplit,'tmp.data_splitT',numT{1}],'...'])
cmd = [fsloutput,'fslcpgeom ',[folder_dataSplit,'tmp.data_splitT',numT{1}],' ',fname_rmse,' -d'];
j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end


	
	
	
mask_brain = 0;

	% compute RMSE within mask
	if mask_brain
		
		j_disp(fname_log,['\nCompute normalized RMSE within brain mask...'])
 
		% mask
		mask = logical(read_avw(fname_mask));
		
		% uncorrected data
		rmse = read_avw(fname_rmse);
		rmse_mask = rmse(mask);
		rmse_mean = mean(rmse_mask);
		rmse_max = max(rmse_mask);
		j_disp(fname_log,['.. Mean RMSE uncorrected: ',num2str(rmse_mean)])
		j_disp(fname_log,['.. Max RMSE uncorrected: ',num2str(rmse_max)])
		
		% corrected data
		rmse = read_avw(fname_rmse_corr);
		rmse_mask = rmse(mask);
		rmse_mean = mean(rmse_mask);
		rmse_max = max(rmse_mask);
		j_disp(fname_log,['.. Mean RMSE corrected: ',num2str(rmse_mean)])
		j_disp(fname_log,['.. Max RMSE corrected: ',num2str(rmse_max)])
		
	end
	
	
% clear temp files
j_disp(fname_log,['\n Clear temp files...'])
cmd = ['rm -rf ',folder_eddy];
j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end


% end
j_disp(fname_log,['\n.. Ended: ',datestr(now)])
j_disp(fname_log,['==========================================================================================================\n'])

