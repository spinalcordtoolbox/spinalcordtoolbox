function param = j_mri_moco_v1(param)
% =========================================================================
% Module that performs motion correction. Usable for anatomical, DTI and
% fMRI data.
% For details on the algorithm see:
% https://docs.google.com/drawings/d/1ndQ68ZK56e2zE5yNEhU9mk_ZCyGSYSMr4X_ARF6rmWc/edit?hl=en_US
% 
% 
% INPUT
% param				structure
%   todo				string		'estimate' | 'apply' | 'both'. NB: 'apply' requires input matrix. Default = 'both'.
%   fname_data			string
%   fname_target		string
%   fname_data_moco		string
%	fname_log			string		fname to log file
%	nt					float		number of volumes
%	nz					float		number of slices
%   slicewise			binary		slice by slice or volume-based motion correction?
%   index				1xn int		indexation that tells which files to correct
%   cost_function		string		'mutualinfo' | 'woods' | 'corratio' | 'normcorr' | 'normmi' | 'leastsquares'. Default is 'normcorr'.
%   flirt_options		string		Additional FLIRT options. Default is ''.
% 
% OUTPUT
% param
% 
%   Example:
%   TODO
%   
%
% TODO
% - for DTI, have a test that checks the mean intensity: if too low, use the registration matrix of the previous volume
% 
% Author: Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% Created: 2011-06-13
% Modified: 2011-06-24
% =========================================================================

% debug if error
dbstop if error

% INITIALIZATIONS
fname_data				= param.fname_data;
fname_data_moco			= param.fname_data_moco;
fname_target			= param.fname_target;
fname_data_splitT		= 'tmp.data_splitT';
if isfield(param,'todo'), todo = param.todo; else todo = 'both'; end
if isfield(param,'fname_log'), fname_log = param.fname_log; else fname_log = 'log_j_mri_moco_v1.txt'; end
if isfield(param,'fsloutput'), fsloutput = param.fsloutput; else fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '; end
if isfield(param,'cost_function'), cost_function = param.cost_function; else cost_function = 'normcorr'; end
if isfield(param,'flirt_options'), flirt_options = param.flirt_options; else flirt_options = ''; end
nt						= param.nt;
nz						= param.nz;

% START FUNCTION
j_disp(fname_log,['\n\n\n=========================================================================================================='])
j_disp(fname_log,['   Running: j_mri_moco_v1'])
j_disp(fname_log,['=========================================================================================================='])
j_disp(fname_log,['\n.. Started: ',datestr(now)])


% split into T dimension
j_disp(fname_log,['Split into T dimension...'])
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

% motion correction module. Loop on T
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
		
		
% =========================================================================
% MOCO
% =========================================================================
		% volume-based or slice-by-slice motion correction?
 		j_disp(fname_log,'Slicewise motion correction?')
		if param.slicewise
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
% 				fname_mat{iT,iZ}
				switch(todo)
					
					case 'estimate'
					cmd = [fsloutput,'flirt -in ',fname_data_splitT_splitZ_num{iT,iZ},' -ref ',fname_data_ref_splitZ_num{iZ},' -omat ',fname_mat{iT,iZ},' -2D -cost ',cost_function,' ',param.flirt_options];

					case 'both'
					cmd = [fsloutput,'flirt -in ',fname_data_splitT_splitZ_num{iT,iZ},' -ref ',fname_data_ref_splitZ_num{iZ},' -out ',fname_data_splitT_splitZ_moco_num{iT,iZ},' -2D -cost ',cost_function,' ',param.flirt_options];
				end
				j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
			end	% iZ
% 			j_disp(fname_log,'---')

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
			cmd = [fsloutput,'flirt -in ',fname_data_splitT_num{iT},' -ref ',fname_target,' -out ',fname_data_splitT_moco_num{iT},' -cost ',cost_function,' ',param.flirt_options];
			j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
		end

% =========================================================================
% END OF MOCO
		
		
		
		
		
	else
		% no
		j_disp(fname_log,'.. no it is not. Don''t correct this volume.')
		copyfile([fname_data_splitT_num{iT},'.nii'],[fname_data_splitT_moco_num{iT},'.nii']);
	end		

end % loop on T

% merge data
j_disp(fname_log,'\nConcatenate along T...')
fname_data_moco_3d = [fname_data_splitT,'_moco_*'];
cmd = [fsloutput,'fslmerge -t ',fname_data_moco,' ',fname_data_moco_3d];
j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
j_disp(fname_log,['.. File created: ',fname_data_moco])


% TODO: CORRECT B-MATRIX DUE TO SUBJECT MOTION
% create new bvecs/bvals files

% Delete temp files
j_disp(fname_log,'\nDelete temporary files...')
delete('tmp.*');
		


j_disp(fname_log,['\n.. Ended: ',datestr(now)])
j_disp(fname_log,['==========================================================================================================\n'])


				