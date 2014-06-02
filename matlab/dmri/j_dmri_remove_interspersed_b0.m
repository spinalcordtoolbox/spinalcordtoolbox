function j_dmri_remove_interspersed_b0(fname_data, fname_bvecs)
% =========================================================================
% Module that removes interspersed b=0.
% 
% INPUT
% fname_data			string
% fname_bvecs			string
% (opt)
%   output_suffix		string. Default='clean'
%	fname_bvals			string. Put '' for no bvals (e.g. DSI or Q-Ball).
%   fname_log
%   outputtype			NIFTI* | NIFTI_GZ
% 
% OUTPUT
% -
%
% Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2011-11-15: created
% 2011-11-22: Fix to speed up fslsplit at the beginning.
% 
% =========================================================================



% INITIALIZATION
if nargin<2, disp('Not enought arguments. Please type "help j_dmri_remove_interspersed_b0".'), return, end
% dbstop if error; % debug if error
if ~exist('opt'), opt = []; end
if isfield(opt,'output_suffix'), output_suffix = opt.output_suffix, else output_suffix = 'clean'; end
if isfield(opt,'fname_bvals'), fname_bvals = opt.fname_bvals, else fname_bvals = ''; end
if isfield(opt,'fname_log'), fname_log = opt.fname_log, else fname_log = 'log_j_dmri_remove_interspersed_b0.txt'; end
if isfield(opt,'outputtype'), outputtype = opt.outputtype, else outputtype = 'NIFTI'; end




% =========================================================================
% START THE SCRIPT
% =========================================================================



% START FUNCTION
j_disp(fname_log,['\n\n\n=========================================================================================================='])
j_disp(fname_log,['   Running: j_dmri_remove_interspersed_b0'])
j_disp(fname_log,['=========================================================================================================='])
j_disp(fname_log,['.. Started: ',datestr(now)])


% output data fname
fname_output = [fname_data,'_',output_suffix];


% Check parameters
j_disp(fname_log,['\nCheck parameters:'])
j_disp(fname_log,['.. Input data:        ',fname_data])
j_disp(fname_log,['.. bvecs file:        ',fname_bvecs])
j_disp(fname_log,['.. Output data:       ',fname_output])
j_disp(fname_log,['.. bvals data:        ',fname_bvals])
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
        fsloutput_temp = ['export FSLOUTPUTTYPE=NIFTI; ']; % if running BASH
elseif strcmp(dmri.shell,'tsh') || strcmp(dmri.shell,'tcsh')
        fsloutput = ['setenv FSLOUTPUTTYPE ',outputtype,'; ']; % if you're running C-SHELL
        fsloutput_temp = ['setenv FSLOUTPUTTYPE NIFTI; ']; % if running BASH
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



% find where are the b=0 images
j_disp(fname_log,['\nIdentify b=0 images...'])
bvecs = textread(fname_bvecs);
index_b0 = [];
for it = 1:nt
	if ~sum(bvecs(it,:)~=[0 0 0])
		index_b0 = cat(1,index_b0,it);
	end
end
nb_b0 = length(index_b0);
j_disp(fname_log,['.. Index of b=0 images (',num2str(nb_b0),'): ',num2str(index_b0')])



% find where are the DW images
j_disp(fname_log,['\nIdentify DW images...'])
index_dwi = [];
for it = 1:nt
	if sum(bvecs(it,:)~=[0 0 0])
		index_dwi = cat(1,index_dwi,it);
	end
end
nb_dwi = length(index_dwi);
j_disp(fname_log,['.. Index of DW images (',num2str(nb_dwi),'): ',num2str(index_dwi')])

	

% split into T dimension
j_disp(fname_log,['\nSplit along T dimension...'])
cmd = [fsloutput_temp,'fslsplit ',fname_data,' tmp.data_splitT'];
j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
numT = j_numbering(nt,4,0);



% Merge DW images
j_disp(fname_log,['\nMerge DW images and add b=0 at the beginning...'])
cmd = [fsloutput,'fslmerge -t ',fname_output];
% add b=0 (take the first one)
cmd = cat(2,cmd,[' tmp.data_splitT',numT{index_b0(1)}]);
% add DW images
for iT = 1:nb_dwi
	cmd = cat(2,cmd,[' tmp.data_splitT',numT{index_dwi(iT)}]);
end
j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
j_disp(fname_log,['.. File created: ',fname_output])



% Clean bvecs/bvals
j_disp(fname_log,['\nClean bvecs/bvals...'])
bvecs_new = [];
if ~isempty(fname_bvals), bvals_new = []; end
% add b=0 bvecs/bvals
bvecs_new = bvecs(index_b0(1),:);
if ~isempty(fname_bvals), bvals_new = dmri.gradients.bvecs(index_b0(1)); end
% add DW bvecs/bvals
for iT = 1:nb_dwi
	bvecs_new = cat(1,bvecs_new,bvecs(index_dwi(iT),:));
	if ~isempty(fname_bvals), bvals_new = cat(1,bvals_new,bvals(index_dwi(iT))); end
end
j_disp(fname_log,['.. New number of directions (including b=0 at the start): ',num2str(size(bvecs_new,1))])
fname_bvecs_new = [fname_bvecs,'_',output_suffix];
if ~isempty(fname_bvals), fname_bvals_new = [fname_bvals,'_',output_suffix]; end
% save files
j_dmri_gradientsWrite(bvecs_new,fname_bvecs_new,'fsl');
if ~isempty(fname_bvals), j_dmri_gradientsWriteBvalue(bvals_new,fname_bvals_new,'fsl'); end
j_disp(fname_log,['.. File created: ',fname_bvecs_new])
if ~isempty(fname_bvals), j_disp(fname_log,['.. File created: ',fname_bvals_new]); end



% remove temp files
j_disp(fname_log,['\nRemove temp files...'])
delete('tmp.*')



% end
j_disp(fname_log,['\n.. Ended: ',datestr(now)])
j_disp(fname_log,['==========================================================================================================\n'])
