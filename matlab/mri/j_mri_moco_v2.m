function param = j_mri_moco_v2(param)
% =========================================================================
% Module that performs motion correction. Usable for anatomical, DTI and
% fMRI data.
% For details on the algorithm see:
% https://docs.google.com/drawings/d/1FoKXYbyFh_q20zsvl_mEcxlUR405gZ4c8DcrUvBxsIM/edit?hl=en_US
% 
% 
% INPUT
% param				structure
%   todo				string		'estimate' | 'apply' | 'estimate_and_apply'. NB: 'apply' requires input matrix. Default = 'estimate_and_apply'.
%   fname_data			string
%   fname_target		string
%   fname_data_moco		string
%   fname_mat			cell		ONLY IF todo=apply. Contains a cell of strings with file name of mat file used by FLIRT to apply correction.
%	fname_log			string		fname to log file
%	nt					float		number of volumes
%	nz					float		number of slices
%   slicewise			binary		slice by slice or volume-based motion correction?
%   index				1xn int		indexation that tells which files to correct
%   cost_function		string		'mutualinfo' | 'woods' | 'corratio' | 'normcorr' | 'normmi' | 'leastsquares'. Default is 'normcorr'.
%   flirt_options		string		Additional FLIRT options. Default is ''.
%   fname_log 			string log file name.
% 
% OUTPUT
% param
% 
%   Example:
%   TODO
%   
%
% TODO
% - manage interspersed for volume-based
% - to need to interpolate sinc when only estimating
% - for DTI, have a test that checks the mean intensity: if too low, use the registration matrix of the previous volume
% 
% Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2011-06-13: Created
% 2011-07-07: Modified
% 2011-10-07: esthetic modifs
% 
% =========================================================================

% debug if error
dbstop if error

% INITIALIZATIONS
fname_data				= param.fname_data;
fname_data_moco			= param.fname_data_moco;
fname_data_splitT		= 'tmp.data_splitT';
if isfield(param,'todo'), todo = param.todo; else todo = 'estimate_and_apply'; end
if isfield(param,'fname_target'), fname_target = param.fname_target; else fname_target = ''; end
if isfield(param,'fname_mat'), fname_mat = param.fname_mat; else fname_mat = ''; end
if isfield(param,'cost_function'), cost_function = param.cost_function; else cost_function = 'normcorr'; end
if isfield(param,'flirt_options'), flirt_options = param.flirt_options; else flirt_options = ''; end
if isfield(param,'fname_log'), fname_log = param.fname_log; else fname_log = 'log_j_mri_moco_v1.txt'; end
if isfield(param,'fsloutput'), fsloutput = param.fsloutput; else fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '; end
nt						= param.nt;
nz						= param.nz;

% START FUNCTION
j_disp(fname_log,['\n\n\n=========================================================================================================='])
j_disp(fname_log,['   Running: j_mri_moco_v2'])
j_disp(fname_log,['=========================================================================================================='])
j_disp(fname_log,['.. Started: ',datestr(now)])

% Check parameters
j_disp(fname_log,['\nCheck parameters:'])
j_disp(fname_log,['.. todo:              ',todo])
j_disp(fname_log,['.. fname_source:      ',fname_data])
j_disp(fname_log,['.. fname_target:      ',fname_target])
j_disp(fname_log,['.. cost_function:     ',cost_function])
j_disp(fname_log,['.. flirt_options:     ',flirt_options])
j_disp(fname_log,['.. fname_target:      ',fname_target])
j_disp(fname_log,['.. Number of volumes: ',num2str(nt)])
j_disp(fname_log,['.. Number of slices:  ',num2str(nz)])

% split into T dimension
j_disp(fname_log,['\nSplit along T dimension...'])
cmd = [fsloutput,'fslsplit ',fname_data,' ',fname_data_splitT];
j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
numT = j_numbering(nt,4,0);

% check if there is an indexation that tells which volume should be corrected
j_disp(fname_log,'\nCheck if there is an index that tells which volume to correct...')
if isfield(param,'index')
	if ~isempty(param.index)
		j_disp(fname_log,'Found!')
	else
		% no indexation. Create one that includes all the volumes.
		j_disp(fname_log,'No indexation found. Create one that includes all the volumes.')
		param.index = (1:nt)';
	end	
else
	% no indexation. Create one that includes all the volumes.
	j_disp(fname_log,'No indexation found. Create one that includes all the volumes.')
	param.index = (1:nt)';
end
j_disp(fname_log,['.. Index of volumes to correct: ',num2str(param.index')])


% volume-based or slice-by-slice motion correction?
j_disp(fname_log,'\nSlicewise motion correction?')
if param.slicewise
	% if slicewise, split target data along Z
	j_disp(fname_log,['Yes! Split target data along Z...'])
	fname_data_ref_splitZ = 'tmp.target_splitZ';
	cmd = [fsloutput,'fslsplit ',fname_target,' ',fname_data_ref_splitZ,' -z'];
	j_disp(fname_log,['>> ',cmd]);
	[status result] = unix(cmd);
	if status, error(result); end
	numZ = j_numbering(nz,4,0);
else
	j_disp(fname_log,['Nope!'])
end





%% MOTION CORRECTION

% Loop on T
j_disp(fname_log,['\n   Motion correction'])
j_disp(fname_log,['-----------------------------------------------'])
j_disp(fname_log,'Loop on iT...')
for iT = 1:nt

	j_disp(fname_log,['\nVolume ',num2str(iT),'/',num2str(nt),':'])
	j_disp(fname_log,['--------------------'])

	% name of volume
	fname_data_splitT_num{iT} = [fname_data_splitT,numT{iT}];
	fname_data_splitT_moco_num{iT} = [fname_data_splitT,'_moco_',numT{iT}];

	% is iT indexed?
	j_disp(fname_log,'Is iT indexed?')
	if find(param.index==iT)
		% yes it is
		j_disp(fname_log,'.. yes it is! Correct this volume.')

		% volume-based or slice-by-slice motion correction?
 		j_disp(fname_log,'Slicewise motion correction?')
		if param.slicewise
			
			% SLICE-WISE MOTION CORRECTION
			% if slicewise, split data along Z
			j_disp(fname_log,['.. Yes!'])
 			j_disp(fname_log,['Split data along Z...'])
			fname_data_splitT_splitZ = [fname_data_splitT_num{iT},'_splitZ'];
			cmd = [fsloutput,'fslsplit ',fname_data_splitT_num{iT},' ',fname_data_splitT_splitZ,' -z'];
			j_disp(fname_log,['>> ',cmd]);
			[status result] = unix(cmd);
			if status, error(result); end

			% loop on Z
			j_disp(fname_log,['Loop on Z...'])
			for iZ = 1:nz
				% motion correction
				% TODO: add more options!
				fname_data_splitT_splitZ_num{iT,iZ} = [fname_data_splitT_splitZ,numZ{iZ}];
				fname_data_splitT_splitZ_moco_num{iT,iZ} = [fname_data_splitT_splitZ_num{iT,iZ},'_moco'];
				fname_data_ref_splitZ_num{iZ} = [fname_data_ref_splitZ,numZ{iZ}];
				switch(todo)
					
					case 'estimate'
	 				fname_mat_estimate{iT,iZ} = ['mat.T',num2str(iT),'_Z',num2str(iZ)];
					cmd = [fsloutput,'flirt -in ',fname_data_splitT_splitZ_num{iT,iZ},' -ref ',fname_data_ref_splitZ_num{iZ},' -omat ',fname_mat_estimate{iT,iZ},' -out ',fname_data_splitT_splitZ_moco_num{iT,iZ},' -2D -cost ',cost_function,' ',param.flirt_options];
% TODO: no need to do sinc interp here
					case 'apply'
					% build file name of input matrix
					fname_mat_apply{iT,iZ} = [fname_mat{iT},'_Z',num2str(iZ)];
					cmd = [fsloutput,'flirt -in ',fname_data_splitT_splitZ_num{iT,iZ},' -ref ',fname_data_ref_splitZ_num{iZ},' -applyxfm -init ',fname_mat_apply{iT,iZ},' -out ',fname_data_splitT_splitZ_moco_num{iT,iZ},' -paddingsize 3 ',param.flirt_options];

					case 'estimate_and_apply'
					cmd = [fsloutput,'flirt -in ',fname_data_splitT_splitZ_num{iT,iZ},' -ref ',fname_data_ref_splitZ_num{iZ},' -out ',fname_data_splitT_splitZ_moco_num{iT,iZ},' -paddingsize 3 -2D -cost ',cost_function,' ',param.flirt_options];
				end
				j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
				
			end	% iZ
			
			% merge into Z dimension
			j_disp(fname_log,['Concatenate along Z...'])
			cmd = [fsloutput,'fslmerge -z ',fname_data_splitT_moco_num{iT}];
			for iZ = 1:nz
				cmd = strcat(cmd,[' ',fname_data_splitT_splitZ_moco_num{iT,iZ}]);
			end
			j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
		
		else
			
			% Volume-based motion correction
			% =======================================================
			j_disp(fname_log,['.. Nope! Volume-based motion correction'])

			switch(todo)

				case 'estimate'
				fname_mat_estimate{iT} = ['mat.T',num2str(iT)];
				cmd = [fsloutput,'flirt -in ',fname_data_splitT_num{iT},' -ref ',fname_target,' -omat ',fname_mat_estimate{iT},' -out ',fname_data_splitT_moco_num{iT},' -cost ',cost_function,' -paddingsize 3 ',param.flirt_options];
% TODO: no need to do sinc interp here

				case 'apply'
				% build file name of input matrix
				fname_mat_apply{iT} = [fname_mat{iT}];
				cmd = [fsloutput,'flirt -in ',fname_data_splitT_num{iT},' -ref ',fname_target,' -applyxfm -init ',fname_mat_apply{iT},' -out ',fname_data_splitT_moco_num{iT},' -paddingsize 3 ',param.flirt_options];

				case 'estimate_and_apply'
				cmd = [fsloutput,'flirt -in ',fname_data_splitT_num{iT},' -ref ',fname_target,' -out ',fname_data_splitT_moco_num{iT},' -cost ',cost_function,' -paddingsize 3 ',param.flirt_options];

			end
			j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
		end

%% END OF MOCO
		
		
		
		
		
	else
		% no
		j_disp(fname_log,'.. no it is not. Don''t correct this volume.')
		copyfile([fname_data_splitT_num{iT},'.nii'],[fname_data_splitT_moco_num{iT},'.nii']);
	end		

end % loop on T

%% merge data
j_disp(fname_log,'\nConcatenate along T...')
fname_data_moco_3d = [fname_data_splitT,'_moco_*'];
cmd = [fsloutput,'fslmerge -t ',fname_data_moco];
for iT = 1:nt
	cmd = cat(2,cmd,[' ',fname_data_splitT,'_moco_',numT{iT}]);
end
j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
j_disp(fname_log,['.. File created: ',fname_data_moco])

% TODO: CORRECT B-MATRIX DUE TO SUBJECT MOTION
% create new bvecs/bvals files

% Delete temp files
j_disp(fname_log,'\nDelete temporary files...')
delete('tmp.*');
		

%% end
j_disp(fname_log,['\n.. Ended: ',datestr(now)])
j_disp(fname_log,['==========================================================================================================\n'])





% function field_var = checkfield(struct,field_str_test,field_str_default,fname_log)
% % =========================================================================
% % check input parameters of this function
% % =========================================================================
% 
% if isfield(struct,field_str)
% 	field_var = struct.todo;
% else field_var = field_str_default;
% end
% j_disp(fname_log,['.. todo = ',param.todo])
				
