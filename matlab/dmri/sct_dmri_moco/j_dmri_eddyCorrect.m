function j_dmri_eddyCorrect(fname_data, fname_bvecs, opt)
% =========================================================================
% 
% Correct Eddy-current distortions using pairs of DW images acquired at
% reversed gradient polarities. Based on the following references:
% TODO
%
% N.B. THIS SCRIPT ASSUMES THE FOLLOWING IMAGE ORIENTATION: 
%	X --> phase. encod.
%	Y --> freq. encod.
%	Z --> slice. encod.
% IF THIS IS NOT THE CASE, THEN CHANGE THE DIMENSION OF YOUR DATA USING
% fslswapdim (or use the option swapXY)
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
%     output_path  string      Default='tmp_eddy';
%   swapXY				0*| 1   Swap X-Y dimension (if your results look worse, you may want to try using this)
%   mask_brain			0*| 1   Create mask automatically using BET and use the mask to register pairs of opposite directions. ## UNSTABLE!
%   min_norm			float.	Minimim norm of two opposite gradient directions (it should be 0 if they are in perfect opposition, but for some reasons, e.g., truncated values, it is note the case. Suggested value=0.001).
%	dof					'TxSx' | 'TxSxKx' | 'TxSxKxKy' | 'Path to your schedule'   Degree of freedom for coregistration of gradient inversed polarity. Tx = Translation along X, Sx = scaling along X, Kx = shearing along X. N.B. data will be temporarily X-Y swapped because FLIRT can only compute shearing parameter along X, not Y
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
%   apply_jacobian		0*| 1 	 Apply Jacobian to correct for intensity modulation due to stretching/expansion
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
% 2011-11-19: Created
% 2011-11-22: Fix to speed up fslsplit at the beginning.
% 2011-11-11: Flag to only estimate transfo (as opposed to estimate+apply)
% 2011-12-17: Enables slice-by-slice correction
% 2011-12-28: Fit transformation parameters (linear least square)
% 2012-01-07: Added Ky parameters to account for z-dependence
% 2012-01-08: Computes RMSE on single shell (for DSI data)
% 2012-01-18: New flag to swap data (in case of axial acquisition)
% 2012-01-19: New flag to mask the brain.
% 2012-01-24: Change RMSE for MSE (minimize noise level). Also, option to smooth MSE map. 
% 2012-01-24: apply Jacobian to correct for intensity modulation due to stretching/expansion
% =========================================================================


% debug if error
dbstop if error

% Check number of arguments
if nargin < 2
	disp('Not enought arguments. Type: help j_dmri_eddyCorrect')
	return
end

% INITIALIZATION
if ~exist('opt'), opt = []; end
if isfield(opt,'dof'), dof = opt.dof; else dof = 'TxSxKx'; end
if isfield(opt,'interpolation'), interpolation = opt.interpolation; else interpolation = 'sinc'; end
if isfield(opt,'cost_function'), cost_function = opt.cost_function; else cost_function = 'normcorr'; end
if isfield(opt,'min_norm'), min_norm = opt.min_norm; else min_norm = 0.001; end
if isfield(opt,'outputtype'), outputtype = opt.outputtype; else outputtype = 'NIFTI'; end
if isfield(opt,'outputsuffix'), outputsuffix = opt.outputsuffix; else outputsuffix = '_eddy'; end
if isfield(opt,'fname_log'), fname_log = opt.fname_log; else fname_log = 'log_j_dmri_eddyCorrect.txt'; end
if isfield(opt,'swapXY'), swapXY = opt.swapXY; else swapXY = 0; end
if isfield(opt,'mask_brain'), mask_brain = opt.mask_brain; else mask_brain = 0; end
if isfield(opt,'split_data'), split_data = opt.split_data; else split_data = 1; end
if isfield(opt,'fname_data_splitT'), fname_data_splitT = opt.fname_data_splitT; else fname_data_splitT = 'tmp_moco.data_splitT'; end
if isfield(opt,'output_path'), output_path = [opt.output_path, 'tmp_eddy/']; else output_path = ['tmp_eddy',filesep]; end
% if isfield(opt,'fname_mat'), fname_mat = opt.fname_mat; else fname_mat = ''; end
if isfield(opt,'folder_mat'), folder_mat = opt.folder_mat; else folder_mat = 'mat_eddy/'; end
% if isfield(opt,'shell'), shell = opt.shell; else shell = ''; end
if isfield(opt,'apply_transfo'), apply_transfo = opt.apply_transfo; else apply_transfo = 1; end
if isfield(opt,'apply_jacobian'), apply_jacobian = opt.apply_jacobian; else apply_jacobian = 0; end
if isfield(opt,'slicewise'), slicewise = opt.slicewise; else slicewise = 1; end
if isfield(opt,'find_transfo'), find_transfo = opt.find_transfo; else find_transfo = 1; end
if isfield(opt,'fit_transfo'), fit_transfo = opt.fit_transfo; else fit_transfo = 0; end
if isfield(opt,'merge_back'), merge_back = opt.merge_back; else merge_back = 1; end
if isfield(opt,'compute_rmse'), compute_rmse = opt.compute_rmse; else compute_rmse = 1; end
if isfield(opt,'rmse_norm_range'), rmse_norm_range = opt.rmse_norm_range; else rmse_norm_range = [0.9 1.1]; end
if isfield(opt,'rmse_smooth'), rmse_smooth = opt.rmse_smooth; else rmse_smooth = 2; end
if isfield(opt,'display_fig'), display_fig = opt.display_fig; else display_fig = 1; end




% =========================================================================
% START THE SCRIPT
% =========================================================================

% START FUNCTION
j_disp(fname_log,['\n\n\n=========================================================================================================='])
j_disp(fname_log,['   Running: j_dmri_eddyCorrect.m'])
j_disp(fname_log,['=========================================================================================================='])
j_disp(fname_log,['.. Started: ',datestr(now)])


% Check parameters
j_disp(fname_log,['\nCheck parameters:'])
j_disp(fname_log,['.. Input data:        ',fname_data])
j_disp(fname_log,['.. bvecs file:        ',fname_bvecs])
j_disp(fname_log,['.. swapXY:            ',num2str(swapXY)])
j_disp(fname_log,['.. mask_brain:        ',num2str(mask_brain)])
j_disp(fname_log,['.. split_data:        ',num2str(split_data)])
j_disp(fname_log,['.. fname_data_splitT: ',fname_data_splitT])
j_disp(fname_log,['.. Degree of freedom: ',dof])
j_disp(fname_log,['.. Interpolation:     ',interpolation])
j_disp(fname_log,['.. Cost function:     ',cost_function])
j_disp(fname_log,['.. Output type:       ',outputtype])
j_disp(fname_log,['.. Output suffixe:    ',outputsuffix])
% j_disp(fname_log,['.. fname_mat:         ',fname_mat])
j_disp(fname_log,['.. folder_mat:        ',folder_mat])
j_disp(fname_log,['.. log file:          ',fname_log])
% j_disp(fname_log,['.. shell:             ',shell])
j_disp(fname_log,['.. slicewise:         ',num2str(slicewise)])
j_disp(fname_log,['.. find_transfo:      ',num2str(find_transfo)])
j_disp(fname_log,['.. fit_transfo:       ',num2str(fit_transfo)])
j_disp(fname_log,['.. apply_transfo:     ',num2str(apply_transfo)])
j_disp(fname_log,['.. apply_jacobian:    ',num2str(apply_jacobian)])
j_disp(fname_log,['.. merge_back:        ',num2str(merge_back)])
j_disp(fname_log,['.. compute_rmse:      ',num2str(compute_rmse)])
j_disp(fname_log,['.. rmse_norm_range:    [',num2str(rmse_norm_range(1)),',',num2str(rmse_norm_range(2)),']'])
j_disp(fname_log,['.. rmse_smooth:       ',num2str(rmse_smooth)])
j_disp(fname_log,['.. display_fig:       ',num2str(display_fig)])



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
	fsloutput = ['export FSLOUTPUTTYPE=',outputtype,'; ']; % if running BASH
	fsloutput_temp = ['export FSLOUTPUTTYPE=NIFTI; '];
elseif strcmp(shell,'tsh') || strcmp(shell,'tcsh') 
	fsloutput = ['setenv FSLOUTPUTTYPE ',outputtype,'; ']; % if you're running T-SHELL
	fsloutput_temp = ['setenv FSLOUTPUTTYPE NIFTI; '];
end
% extension
if strcmp(outputtype,'NIFTI')
	ext = '.nii';
elseif strcmp(outputtype,'NIFTI_GZ')
	ext = '.nii.gz';
end



% Create folder for mat files
if ~exist(folder_mat), mkdir(folder_mat), end
folder_mat_fit = [folder_mat(1:end-1),'_fit',filesep];
if ~exist(output_path), mkdir(output_path), end



% Generate schedule file for FLIRT
j_disp(fname_log,['\nCreate schedule file for FLIRT...'])
schedule_file = [output_path,'schedule_',dof];
switch (dof)
	case 'TxSx'
	fname_schedule = which('j_mri_schedule_TxSx.m');
	
	case 'TxSxKx'
	fname_schedule = which('j_mri_schedule_TxSxKx.m');
		
	case 'TxSxKy'
	fname_schedule = which('j_mri_schedule_TxSxKy.m');
		
	case 'TxSxKxKy'
	fname_schedule = which('j_mri_schedule_TxSxKxKy.m');
		
	case 'TxTyTzSxSySzKxKyKz'
	fname_schedule = which('j_mri_schedule_TxTyTzSxSySzKxKyKz.m');
    
    case 'Tx'
    fname_schedule = which('j_mri_schedule_Tx.m');
    
    otherwise
    schedule_file = [output_path,'schedule'];
    fname_schedule = dof;
    dof='UserTransfo';
    
end
% check if schedule file was found
if isempty(fname_schedule)
	error('Schedule file was not found. Thanks for playing with us.')
end
j_disp(fname_log,['.. Schedule file: ',fname_schedule])
unix(['cp ' fname_schedule ' ' schedule_file]);
j_disp(fname_log,['.. File created (locally): ',schedule_file])



% Swap X-Y dimension (to have X as phase-encoding direction)
if swapXY
	j_disp(fname_log,['\nSwap X-Y dimension (to have X as phase-encoding direction)'])
	fname_data_new = [output_path,'tmp.data_swap'];
	cmd = [fsloutput_temp,'fslswapdim ',fname_data,' -y -x -z ',fname_data_new];
	j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	% update data file name
else
	fname_data_new  = fname_data;
end
j_disp(fname_log,['.. updated data file name: ',fname_data_new])

		

% get data dimensions
j_disp(fname_log,['\nGet dimensions of the data...'])
[~,dims] = read_avw(fname_data_new);
nx = dims(1);
ny = dims(2);
nz = dims(3);
nt = dims(4);
j_disp(fname_log,['.. ',num2str(nx),' x ',num2str(ny),' x ',num2str(nz),' x ',num2str(nt)])
clear img



% split into T dimension
if split_data
	j_disp(fname_log,['\nSplit along T dimension...'])
	cmd = [fsloutput_temp,'fslsplit ',fname_data_new,' ',output_path,'tmp.data_splitT'];
	j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
end
numT = j_numbering(nt,4,0);



% Identify pairs of opposite gradient directions
j_disp(fname_log,['\nIdentify pairs of opposite gradient directions...'])
bvecs = textread(fname_bvecs);
if size(bvecs,1)~=nt
    error(['bvecs file has ' num2str(size(bvecs,1)) ' lines, while data have ' num2str(nt) ' images'])
end
iN = 1;
opposite_gradients = {};
index_identified = [];
index_b0 = []; % b0 index for masking
for iT=1:nt
    if norm(bvecs(iT,:))~=0
        if isempty(find(index_identified==iT))
            for jT=iT+1:nt
                if norm(bvecs(iT,:)+bvecs(jT,:))<min_norm
                    j_disp(fname_log,['.. Opposite gradient for #',num2str(iT),' is: #',num2str(jT)])
                    opposite_gradients{iN} = [iT,jT];
                    index_identified = [index_identified iT jT];
                    iN = iN+1;
                    break
                end
            end
        end
    else
		index_b0 = [index_b0 iT];
        j_disp(fname_log,['.. Opposite gradient for #',num2str(iT),' is: NONE (b=0)'])
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
	fname_b0_merge = [output_path,'b0_merge'];
	cmd = [fsloutput,'fslmerge -t ',fname_b0_merge];
	for iT = 1:nb_b0
		cmd = [cmd,' ',output_path,'tmp.data_splitT',numT{index_b0(iT)}];
	end
	j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	% Average b=0 images
	fname_b0_mean = [output_path,'b0_mean'];
	cmd = [fsloutput,'fslmaths ',fname_b0_merge,' -Tmean ',fname_b0_mean];
	j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	% create mask using BET
	cmd = [fsloutput,'bet2 ',fname_b0_mean,' brain_eddy -m -n -f 0.3'];
	j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	fname_mask = 'brain_eddy_mask';
end



% Slice-wise?
if slicewise

	nb_loops = nz;
	numZ = j_numbering(nz,4,0);
	% build file suffix
	for iZ=1:nz
		file_suffix{iZ} = ['_Z',numT{iZ}];
	end

else
	% volume-based
	nb_loops = 1;
	file_suffix = {''};
end







% nb_oppositeGradients=10





% ================================================================================================================================
%	Find transformation
% ================================================================================================================================
if find_transfo
	for iN=1:nb_oppositeGradients

		i_plus = opposite_gradients{iN}(1);
		i_minus = opposite_gradients{iN}(2);

		% Correct Eddy-Currents for pairs #/#
		j_disp(fname_log,['\nFind affine transformation between volumes #',num2str(i_plus),' and #',num2str(i_minus),' (',num2str(iN),'/',num2str(nb_oppositeGradients),')'])
		j_disp(fname_log,['------------------------------------------------------------------------------------'])

		% volume-based or slice-by-slice correction?
		if slicewise

			% Split volume
			j_disp(fname_log,['Split volumes across Z...'])
			% plus
			fname_plus = [output_path,'tmp.data_splitT',numT{i_plus}];
			fname_plus_Z = [output_path,'tmp.data_splitT',numT{i_plus},'_Z'];
			cmd = [fsloutput_temp,'fslsplit ',fname_plus,' ',fname_plus_Z,' -z'];
			j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
			% minus
			fname_minus = [output_path,'tmp.data_splitT',numT{i_minus}];
			fname_minus_Z = [output_path,'tmp.data_splitT',numT{i_minus},'_Z'];
			cmd = [fsloutput_temp,'fslsplit ',fname_minus,' ',fname_minus_Z,' -z'];
			j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end

		end

		% loop across Z (only one loop if volume-based)
		for iZ=1:nb_loops

% 			% swap X-Y dimensions (because FLIRT can only find shearing along X, and we want Y).
% 			j_disp(fname_log,['Swap X-Y dimensions...'])
 			fname_plus = [output_path,'tmp.data_splitT',numT{i_plus},file_suffix{iZ}];
% 			fname_plus_swap = [fname_plus,'_swap'];
% 			cmd = [fsloutput_temp,'fslswapdim ',fname_plus,' y x z ',fname_plus_swap];
% 			j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
 			fname_minus = [output_path,'tmp.data_splitT',numT{i_minus},file_suffix{iZ}];
% 			fname_minus_swap = [fname_minus,'_swap'];
% 			cmd = [fsloutput_temp,'fslswapdim ',fname_minus,' y x z ',fname_minus_swap];
% 			j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end

			% Find transformation on opposite gradient directions
			j_disp(fname_log,['Find transformation for each pair of opposite gradient directions...'])
			fname_plus_corr = [output_path,'tmp.data_splitT',numT{i_plus},file_suffix{iZ},'_corr_',dof];
			omat = [output_path,'mat__tmp.data_splitT',numT{i_plus},file_suffix{iZ},'_',dof];
			cmd = [fsloutput_temp,'flirt -in ',fname_plus,' -ref ',fname_minus,' -paddingsize 3 -schedule ',schedule_file,' -verbose 2 -omat ',omat,' -cost ',cost_function,' -forcescaling'];
			if mask_brain
				cmd = [cmd, ' -refweight ',fname_mask,' -inweight ',fname_mask];
			end
			j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
			M = textread(omat);
			M = M(1:4,1:4);
			j_disp(fname_log,['.. Transformation matrix: ',mat2str(M)]);
			j_disp(fname_log,['.. Output matrix file: ',omat]);

			% Divide affine transformation by two
			j_disp(fname_log,['Divide affine transformation by two...'])
			A=(M - eye(4))/2;
			% plus
			Mplus = eye(4)+A;
			omat_plus = [folder_mat,'mat.T',num2str(i_plus),'_Z',num2str(iZ),'.txt'];
			fid = fopen(omat_plus,'w');
			fprintf(fid,'%1.6f %1.6f %1.6f %1.6f\n',Mplus');
			fclose(fid);
			j_disp(fname_log,['.. Output matrix file (plus): ',omat_plus]);
			% minus
			Mminus = eye(4)-A;
			omat_minus = [folder_mat,'mat.T',num2str(i_minus),'_Z',num2str(iZ),'.txt'];
			fid = fopen(omat_minus,'w');
			fprintf(fid,'%1.6f %1.6f %1.6f %1.6f\n',Mminus');
			fclose(fid);
			j_disp(fname_log,['.. Output matrix file (minus): ',omat_minus]);
		end
	end
end













% =========================================================================
%	Fit transformation coefficients
% =========================================================================
if fit_transfo
    j_disp(fname_log,['\nFit transformation coefficients'])
	j_disp(fname_log,['------------------------------------------------------------------------------------'])

	% loop across matrices
	j_disp(fname_log,['Extract coefficients...'])
    T = zeros(nb_loops,nt); % Y translation
    S = zeros(nb_loops,nt); % Y scaling
    K = zeros(nb_loops,nt); % XY shearing
    Gx = zeros(nb_loops,nt); % XY shearing
    Gy = zeros(nb_loops,nt); % XY shearing
    Gz = zeros(nb_loops,nt); % XY shearing
    for iT=1:nt
		for iZ=1:nb_loops
			fname_M = [folder_mat,'mat.T',num2str(iT),'_Z',num2str(iZ),'.txt'];
			if exist(fname_M)
				M = textread(fname_M);
				% extract translation parameter
				T(iZ,iT) = M(1,4);
				% extract scaling parameter
				S(iZ,iT) = 1-M(1,1);
				% extract XY shearing parameter
				K(iZ,iT) = M(1,2);
				% extract gradient strength
				Gx(iZ,iT) = (bvecs(iT,1));
				Gy(iZ,iT) = (bvecs(iT,2));
				Gz(iZ,iT) = (bvecs(iT,3));
			end
        end
	end
 
    % fit Ty
	j_disp(fname_log,['Fit Ty...'])
	Tfit = j_dmri_eddyCorrect_fitTransfo(Gx',Gy',Gz',T','Ty',nb_loops,display_fig);
    % fit Sy
	j_disp(fname_log,['Fit Sy...'])
	Sfit = j_dmri_eddyCorrect_fitTransfo(Gx',Gy',Gz',S','Sy',nb_loops,display_fig);
    % fit Ky
	j_disp(fname_log,['Fit Ky...'])
	Kfit = j_dmri_eddyCorrect_fitTransfo(Gx',Gy',Gz',K','Ky',nb_loops,display_fig);
	
	% Write fitted transformation matrices
	j_disp(fname_log,['Write fitted transformation matrices...'])
	if ~exist(folder_mat_fit), mkdir(folder_mat_fit), end
    for iT=1:nt
		for iZ=1:nb_loops
			fname_M = [folder_mat,'mat_T',num2str(iT),'_Z',num2str(iZ),'.txt'];
			if exist(fname_M)
 				% Create Identity matrix
				Mfit = eye(4);
				% update translation parameter
				Mfit(1,4) = Tfit(iZ,iT);
				% extract scaling parameter
				Mfit(1,1) = 1-Sfit(iZ,iT);
				% extract XY shearing parameter
				Mfit(1,2) = Kfit(iZ,iT);
				% Write fitted transformation matrix
				fname_M_fit = [folder_mat_fit,'mat.T',num2str(iT),'_Z',num2str(iZ),'.txt'];
				fid = fopen(fname_M_fit,'w');
				fprintf(fid,'%1.6f %1.6f %1.6f %1.6f\n%1.6f %1.6f %1.6f %1.6f\n%1.6f %1.6f %1.6f %1.6f\n%1.6f %1.6f %1.6f %1.6f',Mfit(1,1),Mfit(1,2),Mfit(1,3),Mfit(1,4),Mfit(2,1),Mfit(2,2),Mfit(2,3),Mfit(2,4),Mfit(3,1),Mfit(3,2),Mfit(3,3),Mfit(3,4),Mfit(4,1),Mfit(4,2),Mfit(4,3),Mfit(4,4));
				fclose(fid);
			end
        end
	end
	
	% update default matrix folder
	folder_mat = folder_mat_fit;
end















% =========================================================================
%	Apply affine transformation
% =========================================================================
if apply_transfo
	j_disp(fname_log,['\nApply affine transformation matrix'])
	j_disp(fname_log,['------------------------------------------------------------------------------------'])
	
	for iN=1:nb_oppositeGradients

		% loop between plus and minus
		for iFile = 1:2
			
			% get i_plus or i_minus index
			i_file = opposite_gradients{iN}(iFile);

			for iZ=1:nb_loops

				if apply_jacobian
					% convert matrix into warping field
					fname = [output_path,'tmp.data_splitT',numT{i_file},file_suffix{iZ}];
					omat = [folder_mat,'mat.T',num2str(i_file),'_Z',num2str(iZ),'.txt'];
					fname_warp = [output_path,'warp_T',numT{i_file},file_suffix{iZ}];
					cmd = [fsloutput_temp,'convertwarp -m ',omat,' -r ',fname,' -o ',fname_warp];
					j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end

					% apply transfo
					fname = [output_path,'tmp.data_splitT',numT{i_file},file_suffix{iZ}];
					fname_corr = [fname,'_corr_',dof,'__div2'];
					fname_warp = [output_path,'warp_T',numT{i_file},file_suffix{iZ}];
					cmd = [fsloutput_temp,'applywarp --in=',fname,' --ref=',fname,' --out=',fname_corr,' --warp=',fname_warp,' --interp=',interpolation, ' --rel'];
					j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end

				else
					% apply transfo
					fname = [output_path,'tmp.data_splitT',numT{i_file},file_suffix{iZ}];
					fname_corr = [fname,'_corr_',dof,'__div2'];
					omat = [folder_mat,'mat.T',num2str(i_file),'_Z',num2str(iZ),'.txt'];
					cmd = [fsloutput_temp,'flirt -in ',fname,' -ref ',fname,' -out ',fname_corr,' -init ',omat,' -applyxfm -paddingsize 3 -interp ',interpolation];
					j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
				end
			end
		end
	end
end
















% ================================================================================================================================
%	Merge back across Z (in case of slicewise correction)
% ================================================================================================================================
if slicewise && find_transfo
	j_disp(fname_log,['\nMerge across Z'])
	j_disp(fname_log,['------------------------------------------------------------------------------------'])
	% loop across direction pairs
	for iN=1:nb_oppositeGradients

		% plus
		i_plus = opposite_gradients{iN}(1);
		fname_plus_corr = [output_path,'tmp.data_splitT',numT{i_plus},'_corr_',dof,'__div2'];
		cmd = [fsloutput_temp,'fslmerge -z ',fname_plus_corr];
		for iZ=1:nz
			fname_plus_Z_corr = [output_path,'tmp.data_splitT',numT{i_plus},file_suffix{iZ},'_corr_',dof,'__div2'];			
			cmd = cat(2,cmd,' ',fname_plus_Z_corr);
		end
		j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end

		% minus
		i_minus = opposite_gradients{iN}(2);
		fname_minus_corr = [output_path,'tmp.data_splitT',numT{i_minus},'_corr_',dof,'__div2'];
		cmd = [fsloutput_temp,'fslmerge -z ',fname_minus_corr];
		for iZ=1:nz
			fname_minus_Z_corr = [output_path,'tmp.data_splitT',numT{i_minus},file_suffix{iZ},'_corr_',dof,'__div2'];			
			cmd = cat(2,cmd,' ',fname_minus_Z_corr);
		end
		j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	end		
end
	



















% ================================================================================================================================
%	RMSE
% ================================================================================================================================
% N.B. Do that in Matlab as opposed to via FSL command, because calculating
% the Squared error and then save it to NIFTI can create overflow error
% because of the signed format of the NIFTI data (i.e., very high value
% would become negative).
if compute_rmse
	
	j_disp(fname_log,['\n\n\nCompute RMSE'])
	j_disp(fname_log,['------------------------------------------------------------------------------------'])

	% Identify directions to use for RMSE
	j_disp(fname_log,['\nIdentify gradient strengh to use for RMSE...'])
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
		fname_plus = [output_path,'tmp.data_splitT',numT{i_plus}];
		if rmse_smooth
			cmd = [fsloutput_temp,'fslmaths ',fname_plus,' -s ',num2str(rmse_smooth),' ',fname_plus,'_smooth'];
			j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
			fname_plus = [fname_plus,'_smooth'];
		end
		[data_plus,dims,scales,bpp,endian] = read_avw(fname_plus);
		% minus
		i_minus = opposite_gradients{rmse_oppositeGradients(iN)}(2);
		fname_minus = [output_path,'tmp.data_splitT',numT{i_minus}];
		if rmse_smooth
			cmd = [fsloutput_temp,'fslmaths ',fname_minus,' -s ',num2str(rmse_smooth),' ',fname_minus,'_smooth'];
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
	fname_rmse = ['MSE_s',num2str(rmse_smooth)];
	j_disp(fname_log,['.. output name: ',fname_rmse])
	save_avw(norm_rmse,fname_rmse,'f',scales(1:3));
	j_disp(fname_log,['\nCopy geometric information from ',[output_path,'tmp.data_splitT',numT{1}],'...'])
	cmd = [fsloutput_temp,'fslcpgeom ',[output_path,'tmp.data_splitT',numT{1}],' ',fname_rmse,' -d'];
	j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end


	% RMSE map of corrected data
	% -------------------------------------------------------------
	se = zeros(nb_oppositeGradients_rmse,nx,ny,nz);
	% loop across direction pairs
	j_disp(fname_log,['\nCompute normalized MSE of corrected data...'])
	for iN=1:nb_oppositeGradients_rmse
		j_disp(fname_log,['.. Pair ',num2str(iN),'/',num2str(nb_oppositeGradients_rmse)])
		% plus
		i_plus = opposite_gradients{rmse_oppositeGradients(iN)}(1);
		fname_plus = [output_path,'tmp.data_splitT',numT{i_plus},'_corr_',dof,'__div2'];
		if rmse_smooth
			cmd = [fsloutput_temp,'fslmaths ',fname_plus,' -s ',num2str(rmse_smooth),' ',fname_plus,'_smooth'];
			j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
			fname_plus = [fname_plus,'_smooth'];
		end
		[data_plus ,dims,scales,bpp,endian] = read_avw(fname_plus);
		% minus
		i_minus = opposite_gradients{rmse_oppositeGradients(iN)}(2);
		fname_minus = [output_path,'tmp.data_splitT',numT{i_minus},'_corr_',dof,'__div2'];
		if rmse_smooth
			cmd = [fsloutput_temp,'fslmaths ',fname_minus,' -s ',num2str(rmse_smooth),' ',fname_minus,'_smooth'];
			j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
			fname_minus = [fname_minus,'_smooth'];
		end
		[data_minus ,dims,scales,bpp,endian] = read_avw(fname_minus);
		% compute squared error
		se(1,:,:,:) = (data_plus - data_minus).^2;
	end
	% compute normalized RMSE
	norm_rmse = squeeze(mean(se,1))/nb_oppositeGradients_rmse;
	% save volume
	fname_rmse_corr = ['MSE_s',num2str(rmse_smooth),outputsuffix];
	j_disp(fname_log,['.. output name: ',fname_rmse_corr])
	save_avw(norm_rmse,fname_rmse_corr,'f',scales(1:3));
	j_disp(fname_log,['\nCopy geometric information from ',[output_path,'tmp.data_splitT',numT{1}],'...'])
	cmd = [fsloutput,'fslcpgeom ',[output_path,'tmp.data_splitT',numT{1}],' ',fname_rmse_corr,' -d'];
	j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	
	
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
	% RMSE percentage of uncorrected Vs. corrected data
	% -------------------------------------------------------------
% 	j_disp(fname_log,['\nRMSE percentage of uncorrected Vs. corrected data...'])
% 	fname_rmse_corr_perc = ['MSE_corrPercent_',dof];
% 	cmd = [fsloutput_temp,'fslmaths ',fname_rmse,' -sub ',fname_rmse_corr,' -div ',fname_rmse,' -mul -100 ',fname_rmse_corr_perc];
% 	j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end	
% 	j_disp(fname_log,['.. File created: ',fname_rmse_corr_perc]);
end











	



% ================================================================================================================================
%	Merge files back
% ================================================================================================================================
if apply_transfo & merge_back
	
	% Merge back across T
	j_disp(fname_log,['\nMerge back across T...'])
	fname_data_corr = [fname_data_new,outputsuffix];
	% check if nt is big (might create problems, so in that case split the merging process)
	if (nt < 1000)
		% nt is not big
		cmd = [fsloutput,'fslmerge -t ',fname_data_corr];
		for iT=1:nt
			if ~isempty(dir([output_path,'tmp.data_splitT',numT{iT},'_corr_',dof,'__div2.*']))
				% volume has been corrected (so it exists)
				fname_data_corr_3d = [output_path,'tmp.data_splitT',numT{iT},'_corr_',dof,'__div2'];
			else
				% volume has NOT been corrected (so it DOES NOT exist, e.g., b=0 file)
				fname_data_corr_3d = [output_path,'tmp.data_splitT',numT{iT}];
			end
			cmd = cat(2,cmd,' ',fname_data_corr_3d);
		end
		j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	else
		
		% nt is big
		iTsub = 1;
		cmd_sub = [fsloutput,'fslmerge -t ',fname_data_corr,'_sub',num2str(iTsub)];
% 		cmd = [fsloutput,'fslmerge -t ',fname_data_corr];
		iTbig = 1;
		for iT=1:nt
			if ~isempty(dir([output_path,'tmp.data_splitT',numT{iT},'_corr_',dof,'__div2.*']))
				% volume has been corrected (so it exists)
				fname_data_corr_3d = [output_path,'tmp.data_splitT',numT{iT},'_corr_',dof,'__div2'];
			else
				% volume has NOT been corrected (so it DOES NOT exist, e.g., b=0 file)
				fname_data_corr_3d = [output_path,'tmp.data_splitT',numT{iT}];
			end
			cmd_sub = cat(2,cmd_sub,' ',fname_data_corr_3d);
			if iTbig == 1000
				% output the subparts of the merge
				j_disp(fname_log,['>> ',cmd_sub]); [status result] = unix(cmd_sub); if status, error(result); end
				% increment
				iTsub = iTsub + 1;
				% reinitialize
				cmd_sub = [fsloutput,'fslmerge -t ',fname_data_corr,'_sub',num2str(iTsub)];
				iTbig = 1;
			end
			iTbig = iTbig + 1;
		end
		% output the last subpart of the merge
		j_disp(fname_log,['>> ',cmd_sub]); [status result] = unix(cmd_sub); if status, error(result); end
		% merge the subparts of the merge
		cmd = [fsloutput,'fslmerge -t ',fname_data_corr];		
		for iTbig=1:iTsub
			cmd = cat(2,cmd,' ',fname_data_corr,'_sub',num2str(iTbig));
		end
		j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
		% remove temp files
		for iTbig=1:iTsub
			cmd = ['rm ',fname_data_corr,'_sub',num2str(iTbig),'.*'];
			j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
		end
	end

	% Swap back X-Y dimensions
	if swapXY
		fname_data_final = [fname_data,outputsuffix];
		j_disp(fname_log,['\nSwap back X-Y dimensions'])
		cmd = [fsloutput_temp,'fslswapdim ',fname_data_corr,' -y -x -z ',fname_data_final];
		j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	else
		fname_data_final = fname_data_corr;
% 		% don't swap back, just move the data to final folder
% 		cmd = ['mv ',fname_data_corr,' ',fname_data_final];
% 		j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	end
	j_disp(fname_log,['.. File created: ',fname_data_final])
end




% clear temp files
j_disp(fname_log,['\n Clear temp files...'])
cmd = ['rm -rf ',output_path];
j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end


% end
j_disp(fname_log,['\n.. Ended: ',datestr(now)])
j_disp(fname_log,['==========================================================================================================\n'])



function mat = gauss2d(dims, sigma, center)
[R,C] = ndgrid(1:dims(1), 1:dims(2));
mat = gaussC(R,C, sigma, center);

function val = gaussC(x, y, sigma, center)
xc = center(1);
yc = center(2);
exponent = ((x-xc).^2./(2*sigma(1)) + (y-yc).^2./(2*sigma(2)));
val       = (exp(-exponent));




% OLD CODE
% 	
% 	Tfit = zeros(3,nb_loops);
% 	x = Gx'; y = T'; ind_nonzero = find(x(:,1)); 
% 	for iZ=1:nb_loops
% 		Tfit(1,iZ) = inv(x(ind_nonzero,iZ)'*x(ind_nonzero,iZ))*x(ind_nonzero,iZ)'*y(ind_nonzero,iZ);
% 	end
%  	h_fig = figure;
% 	subplot(2,2,1)
% 	for iZ=1:nb_loops
% 		plot(x(ind_nonzero,iZ),y(ind_nonzero,iZ),'.','markersize',15,'color',[(iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gx'), ylabel('Ty (voxel)'), hold on
% 		plot(x(ind_nonzero,iZ),x(ind_nonzero,iZ)*Tfit(1,iZ),'-','color',[1 (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gx'), ylabel('Ty (voxel)'), hold on
% 	end
% 	grid
% 	
% 	x = Gy'; y = T'; ind_nonzero = find(x(:,1));
% 	for iZ=1:nb_loops
% 		Tfit(2,iZ) = inv(x(ind_nonzero,iZ)'*x(ind_nonzero,iZ))*x(ind_nonzero,iZ)'*y(ind_nonzero,iZ);
% 	end
%     subplot(2,2,2)
% 	for iZ=1:nb_loops
% 		plot(x(ind_nonzero,iZ),y(ind_nonzero,iZ),'.','markersize',15,'color',[(iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gy'), ylabel('Ty (voxel)'), hold on
% 		plot(x(ind_nonzero,iZ),x(ind_nonzero,iZ)*Tfit(2,iZ),'-','color',[1 (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gx'), ylabel('Ty (voxel)'), hold on
% 	end
% 	grid
% 	
% 	x = Gz'; y = T'; ind_nonzero = find(x(:,1));
% 	for iZ=1:nb_loops
% 		Tfit(3,iZ) = inv(x(ind_nonzero,iZ)'*x(ind_nonzero,iZ))*x(ind_nonzero,iZ)'*y(ind_nonzero,iZ);
% 	end
% 
%     subplot(2,2,3)
% 	for iZ=1:nb_loops
% 		plot(x(ind_nonzero,iZ),y(ind_nonzero,iZ),'.','markersize',15,'color',[(iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gz'), ylabel('Ty (voxel)'), hold on
% 		plot(x(ind_nonzero,iZ),x(ind_nonzero,iZ)*Tfit(3,iZ),'-','color',[1 (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gx'), ylabel('Ty (voxel)'), hold on
% 	end
% 	grid
% 	print(h_fig,'-dpng','eddyCorr_Ty.png')
% 
% 	% Plot z-dependence
% 	figure, plot(Tfit','linewidth',2), grid
% 	legend('Gx','Gy','Gz')
% 	xlabel('Z')
% 	ylabel('Ty_{fit}')
%  
% 	
% 	
% 	
% 	
%     % plot Sy
% 	h_fig = figure;
% 	x = Gx'; y = S'; ind_nonzero = find(x(:,1));
%     subplot(2,2,1)
% 	for iZ=1:nb_loops
% 		plot(x(ind_nonzero,iZ),y(ind_nonzero,iZ),'.','markersize',15,'color',[(iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gx'), ylabel('Sy (voxel)'), hold on
% 	end
% 	grid
% 	x = Gy'; y = S'; ind_nonzero = find(x(:,1));
%     subplot(2,2,2)
% 	for iZ=1:nb_loops
% 		plot(x(ind_nonzero,iZ),y(ind_nonzero,iZ),'.','markersize',15,'color',[(iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gy'), ylabel('Sy (voxel)'), hold on
% 	end
% 	grid
% 	x = Gz'; y = S'; ind_nonzero = find(x(:,1));
%     subplot(2,2,3)
% 	for iZ=1:nb_loops
% 		plot(x(ind_nonzero,iZ),y(ind_nonzero,iZ),'.','markersize',15,'color',[(iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gz'), ylabel('Sy (voxel)'), hold on
% 	end
% 	grid
% 	print(h_fig,'-dpng','eddyCorr_Sy.png')
%  
% 	% Plot Kxy
% 	h_fig = figure;
% 	y = Kxy'; 
% 	x = Gx'; ind_nonzero = find(x(:,1));
%     subplot(2,2,1)
% 	for iZ=1:nb_loops
% 		plot(x(ind_nonzero,iZ),y(ind_nonzero,iZ),'.','markersize',15,'color',[(iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gx'), ylabel('Kxy (voxel)'), hold on
% 	end
% 	grid
% 	x = Gy'; ind_nonzero = find(x(:,1));
%     subplot(2,2,2)
% 	for iZ=1:nb_loops
% 		plot(x(ind_nonzero,iZ),y(ind_nonzero,iZ),'.','markersize',15,'color',[(iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gy'), ylabel('Kxy (voxel)'), hold on
% 	end
% 	grid
% 	x = Gz'; ind_nonzero = find(x(:,1));
%     subplot(2,2,3)
% 	for iZ=1:nb_loops
% 		plot(x(ind_nonzero,iZ),y(ind_nonzero,iZ),'.','markersize',15,'color',[(iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gz'), ylabel('Kxy (voxel)'), hold on
% 	end
% 	grid
%  	print(h_fig,'-dpng','eddyCorr_Kxy.png')
%     
% 	% Plot Kxz
% 	h_fig = figure;
% 	y = Kxz'; 
% 	x = Gx'; ind_nonzero = find(x(:,1));
%     subplot(2,2,1)
% 	for iZ=1:nb_loops
% 		plot(x(ind_nonzero,iZ),y(ind_nonzero,iZ),'.','markersize',15,'color',[(iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gx'), ylabel('Kxz (voxel)'), hold on
% 	end
% 	grid
% 	x = Gy'; ind_nonzero = find(x(:,1));
%     subplot(2,2,2)
% 	for iZ=1:nb_loops
% 		plot(x(ind_nonzero,iZ),y(ind_nonzero,iZ),'.','markersize',15,'color',[(iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gy'), ylabel('Kxz (voxel)'), hold on
% 	end
% 	grid
% 	x = Gz'; ind_nonzero = find(x(:,1));
%     subplot(2,2,3)
% 	for iZ=1:nb_loops
% 		plot(x(ind_nonzero,iZ),y(ind_nonzero,iZ),'.','markersize',15,'color',[(iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gz'), ylabel('Kxz (voxel)'), hold on
% 	end
% 	grid
%  	print(h_fig,'-dpng','eddyCorr_Kxz.png')
%     
% 	% Plot Kyz
% 	h_fig = figure;
% 	y = Kyz'; 
% 	x = Gx'; ind_nonzero = find(x(:,1));
%     subplot(2,2,1)
% 	for iZ=1:nb_loops
% 		plot(x(ind_nonzero,iZ),y(ind_nonzero,iZ),'.','markersize',15,'color',[(iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gx'), ylabel('Kyz (voxel)'), hold on
% 	end
% 	grid
% 	x = Gy'; ind_nonzero = find(x(:,1));
%     subplot(2,2,2)
% 	for iZ=1:nb_loops
% 		plot(x(ind_nonzero,iZ),y(ind_nonzero,iZ),'.','markersize',15,'color',[(iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gy'), ylabel('Kyz (voxel)'), hold on
% 	end
% 	grid
% 	x = Gz'; ind_nonzero = find(x(:,1));
%     subplot(2,2,3)
% 	for iZ=1:nb_loops
% 		plot(x(ind_nonzero,iZ),y(ind_nonzero,iZ),'.','markersize',15,'color',[(iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gz'), ylabel('Kyz (voxel)'), hold on
% 	end
% 	grid
%  	print(h_fig,'-dpng','eddyCorr_Kyz.png')
    
