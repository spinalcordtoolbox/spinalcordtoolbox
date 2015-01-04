function j_dmri_reorganize_data(fname_data, fname_bvecs_source, fname_bvecs_target)
% =========================================================================
% Module that reorganizes diffusion-weighted data to match a given qvecs
% (or bvecs) file, OR to order by bvals (using bvals file or schemefile)
% 
% 
% INPUT
% fname_data			string
% fname_bvecs_source	string
% fname_bvecs_target	string
% (opt)
%   output_suffix		string. Default='ordered'
%	fname_bvals			string. Put '' for no bvals (e.g. DSI or Q-Ball).
%   fname_log
%   outputtype			NIFTI | NIFTI_GZ
%   bash				'bash'* | 'tsh'
% 
% OUTPUT
% -
%
% Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2011-11-29: created
% 
% =========================================================================



% INITIALIZATION
if nargin<3, help j_dmri_reorganize_data, return, end
% dbstop if error; % debug if error
if ~exist('opt'), opt = []; end
if isfield(opt,'output_suffix'), output_suffix = opt.output_suffix; else output_suffix = 'ordered'; end
if isfield(opt,'fname_log'), fname_log = opt.fname_log; else fname_log = 'log_j_dmri_reorganize_data.txt'; end
if isfield(opt,'outputtype'), outputtype = opt.outputtype; else outputtype = 'NIFTI'; end
if isfield(opt,'shell'), shell = opt.shell; else shell = 'bash'; end

min_norm			= 0.001;


% =========================================================================
% START THE SCRIPT
% =========================================================================



% START FUNCTION
j_disp(fname_log,['\n\n\n=========================================================================================================='])
j_disp(fname_log,['   Running: j_dmri_reorganize_data'])
j_disp(fname_log,['=========================================================================================================='])
j_disp(fname_log,['.. Started: ',datestr(now)])


% output data fname
fname_output = [fname_data,'_',output_suffix];


% Check parameters
j_disp(fname_log,['\nCheck parameters:'])
j_disp(fname_log,['.. Input data:        ',fname_data])
j_disp(fname_log,['.. bvecs source:      ',fname_bvecs_source])
j_disp(fname_log,['.. bvecs target:      ',fname_bvecs_target])
j_disp(fname_log,['.. Output data:       ',fname_output])
j_disp(fname_log,['.. Log file:          ',fname_log])
j_disp(fname_log,['.. Output extension:  ',outputtype])



% Find which SHELL is running
j_disp(fname_log,['\nFind which SHELL is running...'])
[status result] = unix('echo $0');
if ~isempty(findstr(result,'bash'))
        dmri.shell = 'bash';
elseif ~isempty(findstr(result,'tsh'))
        dmri.shell = 'tsh';
elseif ~isempty(findstr(result,'tcsh'))
        dmri.shell = 'tcsh';
else    
        j_disp(dmri.log,['.. Failed to identify shell. Using default.'])
        dmri.shell = 'bash';
end     
j_disp(fname_log,['.. Running: ',dmri.shell])



% FSL output
if strcmp(dmri.shell,'bash')
        fsloutput = ['export FSLOUTPUTTYPE=',outputtype,'; ']; % if running BASH
elseif strcmp(dmri.shell,'tsh') || strcmp(dmri.shell,'tcsh')
        fsloutput = ['setenv FSLOUTPUTTYPE ',outputtype,'; ']; % if you're running C-SHELL
else
        error('Check SHELL field.')
end
% extension
if strcmp(outputtype,'NIFTI')
	ext = '.nii';
elseif strcmp(outputtype,'NIFTI_GZ')
	ext = '.nii.gz';
end



% get data dimensions
fprintf('\n');
j_disp(fname_log,'Get dimensions of the data...')
cmd = ['fslsize ',fname_data]; [status result] = unix(cmd); if status, error(result); end
dims = j_mri_getDimensions(result);
nx = dims(1);
ny = dims(2);
nz = dims(3);
nt = dims(4);
numT = j_numbering(nt,4,0);
j_disp(fname_log,['.. ',num2str(nx),' x ',num2str(ny),' x ',num2str(nz),' x ',num2str(nt)])
clear img



% Open bvecs source
j_disp(fname_log,['\nOpen bvecs source...'])
bvecs_source = textread(fname_bvecs_source);



% Open bvecs target
j_disp(fname_log,['\nOpen bvecs target...'])
bvecs_target = textread(fname_bvecs_target);



% Find look-up correspondance table
j_disp(fname_log,['\nFind look-up table between Source and Target...'])
nb_dirs = size(bvecs_source,1);
nb_dirs_targ = size(bvecs_target,1);
for iT=1:nb_dirs
	for jT=1:nb_dirs_targ
		if norm(bvecs_source(iT,:)-bvecs_target(jT,:)) < min_norm
			j_disp(fname_log,['.. Source #',num2str(iT),' corresponds to Target #',num2str(jT)])
			index_target(jT) = iT;
			% remove value in target (to avoid double picking)
			bvecs_target(jT,:) = [10 10 10];
			break
		end
	end
end



% split into T dimension
j_disp(fname_log,['\nSplit along T dimension...'])
cmd = [fsloutput,'fslsplit ',fname_data,' tmp.data_splitT'];
j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
numT = j_numbering(nt,4,0);



% Re-order and merge back
j_disp(fname_log,['\nRe-order and merge back...'])
cmd = [fsloutput,'fslmerge -t ',fname_output];
% add DW images
for iT = 1:nb_dirs_targ
	cmd = cat(2,cmd,[' tmp.data_splitT',numT{index_target(iT)}]);
end
j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
j_disp(fname_log,['.. File created: ',fname_output])



% remove temp files
j_disp(fname_log,['\nRemove temp files...'])
cmd = ['rm tmp.*'];
j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end



% end
j_disp(fname_log,['\n.. Ended: ',datestr(now)])
j_disp(fname_log,['==========================================================================================================\n'])
