function param = j_mri_moco_v3(param)
% =========================================================================
% Module that performs motion correction. Usable for anatomical, DTI and
% fMRI data.
% For details on the algorithm see:
% https://docs.google.com/drawings/d/1FoKXYbyFh_q20zsvl_mEcxlUR405gZ4c8DcrUvBxsIM/edit?hl=en_US
% 
% 
% INPUT
% param				structure
%
% MANDATORY
%   todo				string		'estimate' | 'apply' | 'estimate_and_apply'. NB: 'apply' requires input matrix. Default = 'estimate_and_apply'.
%   fname_data			string
%   fname_target		string
%   fname_data_moco		string
%	nt					float		number of volumes
%	nz					float		number of slices
%
% FACULTATIVE
%	fname_folder_mat	string		folder where to put mat files. Default='mat/'
%   fname_mat			cell		ONLY IF todo=apply. Contains a cell of strings with file name of mat file used by FLIRT to apply correction.
%	split_data			binary		0 | 1*. If data are already splitted, then indicated the file name in the following flag.
%	fname_data_splitT	string		Default='tmp_moco.data_splitT'
%	fname_log			string		fname to log file
%   slicewise			binary		slice by slice or volume-based motion correction?
%   index				1xn int		indexation that tells which files to correct
%   cost_function		string		'mutualinfo' | 'woods' | 'corratio' | 'normcorr' | 'normmi' | 'leastsquares'. Default is 'normcorr'.
%   flirt_options		string		Additional FLIRT options. E.g., '-interp sinc'. Default is ''.
%   merge_back			binary		0 | 1*. Merge data back after moco?
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
% - no need to interpolate sinc when only estimating
% - for DTI, have a test that checks the mean intensity: if too low, use the registration matrix of the previous volume
% 
% Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2011-06-13: Created
% 2011-07-07: Modified
% 2011-10-07: esthetic modifs
% 2011-12-04: Only applies. Create a mat/ folder to put all the junk.
% 2011-12-11: Add a flag to avoid merging data after moco
% 
% =========================================================================

% debug if error
dbstop if error



% INITIALIZATIONS
fname_data				= param.fname_data;
fname_data_moco			= param.fname_data_moco;
if isfield(param,'todo'), todo = param.todo; else todo = 'estimate_and_apply'; end
if isfield(param,'fname_target'), fname_target = param.fname_target; else fname_target = ''; end
if isfield(param,'fname_mat'), fname_mat = param.fname_mat; else fname_mat = ''; end
if isfield(param,'folder_mat'), folder_mat = param.folder_mat; else folder_mat = 'mat/'; end
if isfield(param,'split_data'), split_data = param.split_data; else split_data = 1; end
if isfield(param,'fname_data_splitT'), fname_data_splitT = param.fname_data_splitT; else fname_data_splitT = 'tmp_moco.data_splitT'; end
% if isfield(param,'delete_tmp_files'), delete_tmp_files = param.delete_tmp_files; else delete_tmp_files = 1; end
if isfield(param,'cost_function'), cost_function = param.cost_function; else cost_function = 'normcorr'; end
if isfield(param,'flirt_options'), flirt_options = param.flirt_options; else flirt_options = ''; end
if isfield(param,'fname_log'), fname_log = param.fname_log; else fname_log = 'log_j_mri_moco_v1.txt'; end
if isfield(param,'fsloutput'), fsloutput = param.fsloutput; else fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '; end
if isfield(param,'merge_back'), merge_back = param.merge_back; else merge_back = 1; end
if isfield(param,'nt'), nt = param.nt; else nt = 1; end
if isfield(param,'nz'), nz = param.nz; else nz = 1; end
if isfield(param,'slicewise'), slicewise = param.slicewise; else slicewise = 0; end


% START FUNCTION
j_disp(fname_log,['\n\n\n=========================================================================================================='])
j_disp(fname_log,['   Running: j_mri_moco_v3'])
j_disp(fname_log,['=========================================================================================================='])
j_disp(fname_log,['.. Started: ',datestr(now)])



% Check parameters
j_disp(fname_log,['\nCheck parameters:'])
j_disp(fname_log,['.. todo:              ',todo])
j_disp(fname_log,['.. fname_source:      ',fname_data])
j_disp(fname_log,['.. fname_target:      ',fname_target])
j_disp(fname_log,['.. fname_mat:         ',fname_mat])
j_disp(fname_log,['.. folder_mat:        ',folder_mat])
j_disp(fname_log,['.. split_data:        ',num2str(split_data)])
j_disp(fname_log,['.. fname_data_splitT: ',fname_data_splitT])
j_disp(fname_log,['.. cost_function:     ',cost_function])
j_disp(fname_log,['.. flirt_options:     ',flirt_options])
j_disp(fname_log,['.. Number of volumes: ',num2str(nt)])
j_disp(fname_log,['.. Number of slices:  ',num2str(nz)])
j_disp(fname_log,['.. Merge data back:   ',num2str(merge_back)])



% split into T dimension
if split_data
	j_disp(fname_log,['\nSplit along T dimension...'])
	cmd = [fsloutput,'fslsplit ',fname_data,' ',fname_data_splitT];
	j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
end
numT = j_numbering(nt,4,0);



% Create folder for mat files
if ~exist(folder_mat), mkdir(folder_mat), end



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
if slicewise
	% if slicewise, split target data along Z
	j_disp(fname_log,['Yes! Split target data along Z...'])
	fname_data_ref_splitZ = 'tmp_moco.target_splitZ';
	cmd = [fsloutput,'fslsplit ',fname_target,' ',fname_data_ref_splitZ,' -z'];
	j_disp(fname_log,['>> ',cmd]);
	[status result] = unix(cmd);
	if status, error(result); end
	numZ = j_numbering(nz,4,0);
else
	j_disp(fname_log,['Nope!'])
end





% MOTION CORRECTION

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
%  		j_disp(fname_log,'Slicewise motion correction?')
		if slicewise
			
			% SLICE-WISE MOTION CORRECTION
			% if slicewise, split data along Z
% 			j_disp(fname_log,['.. Yes!'])
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
					cmd = [fsloutput,'flirt -in ',fname_data_splitT_splitZ_num{iT,iZ},' -ref ',fname_data_ref_splitZ_num{iZ},' -omat ',fname_mat_estimate{iT,iZ},' -out ',fname_data_splitT_splitZ_moco_num{iT,iZ},' -2D -cost ',cost_function,' ',flirt_options];
% TODO: no need to do sinc interp here
					case 'apply'
					% build file name of input matrix
					fname_mat_apply{iT,iZ} = [fname_mat{iT},'_Z',num2str(iZ)];
					cmd = [fsloutput,'flirt -in ',fname_data_splitT_splitZ_num{iT,iZ},' -ref ',fname_data_ref_splitZ_num{iZ},' -applyxfm -init ',fname_mat_apply{iT,iZ},' -out ',fname_data_splitT_splitZ_moco_num{iT,iZ},' -paddingsize 3 ',flirt_options];

					case 'estimate_and_apply'
					cmd = [fsloutput,'flirt -in ',fname_data_splitT_splitZ_num{iT,iZ},' -ref ',fname_data_ref_splitZ_num{iZ},' -out ',fname_data_splitT_splitZ_moco_num{iT,iZ},' -paddingsize 3 -2D -cost ',cost_function,' ',flirt_options];
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
% 			j_disp(fname_log,['.. Nope! Volume-based motion correction'])

			switch(todo)

				case 'estimate'
				fname_mat_estimate{iT} = [folder_mat,'mat.T',numT{iT}];
				cmd = [fsloutput,'flirt -in ',fname_data_splitT_num{iT},' -ref ',fname_target,' -omat ',fname_mat_estimate{iT},' -cost ',cost_function,' ',flirt_options];

				case 'apply'
				% build file name of input matrix
				fname_mat_apply{iT} = [folder_mat,'mat.T',numT{iT}];
				cmd = [fsloutput,'flirt -in ',fname_data_splitT_num{iT},' -ref ',fname_target,' -applyxfm -init ',fname_mat_apply{iT},' -out ',fname_data_splitT_moco_num{iT},' -paddingsize 3 ',flirt_options];

				case 'estimate_and_apply'
				cmd = [fsloutput,'flirt -in ',fname_data_splitT_num{iT},' -ref ',fname_target,' -out ',fname_data_splitT_moco_num{iT},' -cost ',cost_function,' -paddingsize 3 ',flirt_options];

			end
			j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
		end

% END OF MOCO
		
		
		
		
		
	else
		% no
		j_disp(fname_log,'.. no it is not. Don''t correct this volume.')
		copyfile([fname_data_splitT_num{iT},'.nii'],[fname_data_splitT_moco_num{iT},'.nii']);
	end		

end % loop on T

% merge data
if ~strcmp(todo,'estimate')
	if merge_back
		j_disp(fname_log,'\n\nMerge data back along T...')
		fname_data_moco_3d = [fname_data_splitT,'_moco_*'];
		cmd = [fsloutput,'fslmerge -t ',fname_data_moco];
		for iT = 1:nt
			cmd = cat(2,cmd,[' ',fname_data_splitT,'_moco_',numT{iT}]);
		end
		j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
		j_disp(fname_log,['.. File created: ',fname_data_moco])
	end
end


% Delete temp files
j_disp(fname_log,'\nDelete temporary files...')
delete('tmp_moco.*');



% end
j_disp(fname_log,['\n.. Ended: ',datestr(now)])
j_disp(fname_log,['==========================================================================================================\n'])


