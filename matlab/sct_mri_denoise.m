function sct = sct_mri_denoise(sct)
% =========================================================================
% 
% Denoise data using Pierrick Coupé algorithm.
% 
% 
% INPUT
% -------------------------------------------------------------------------
% (opt)
%   fname_in			string
%   (fname_out)			string
%   (fname_log)			string
% 
% -------------------------------------------------------------------------
%
% OUTPUT
% -------------------------------------------------------------------------
% 
% -------------------------------------------------------------------------
% 
%   Example
%   j_mri_denoise
%
%
% Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2013-01-25: Created
%
% =========================================================================

% PARAMETERS


% Check number of arguments
if nargin < 1
	disp('Not enought arguments. Type: help sct_mri_denoise')
	return
end

% INITIALIZATION
dbstop if error; % debug if error
if ~exist('sct'), sct = []; end


fname_in = [sct.output_path,'anat',filesep,sct.anat.file];
fname_out = [fname_in,'_denoised'];
fname_log = 'log_sct_mri_denoise.txt';
outputtype = 'NIFTI';

% START FUNCTION
j_disp(fname_log,['\n\n\n=========================================================================================================='])
j_disp(fname_log,['   Running: sct_mri_denoise'])
j_disp(fname_log,['=========================================================================================================='])
j_disp(fname_log,['.. Started: ',datestr(now)])

% Check parameters
j_disp(fname_log,['\nCheck parameters:'])


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
	fsloutput = ['set FSLOUTPUTTYPE=',outputtype,'; ']; % if you're running T-SHELL
	fsloutput_temp = ['setenv FSLOUTPUTTYPE NIFTI; '];
end

 

% Open image
[ima,dims,scales,bpp,endian] = read_avw(fname_in);

s=size(ima);
mini = min(ima(:));
ima = ima + abs(mini);


%params
level = 10;
M=3;
alpha=1;


j_disp(fname_log,['\nFiltering...'])

% Filtering with Su parameters: small patch
j_disp(fname_log,'.. Filtering with Su parameters: small patch')
tic,
h=level;
fima1=onlm(ima,M,alpha,h);
fima1=fima1 - abs(mini);
t1(1)=toc;

% Filtering with So parameters: big patch 
j_disp(fname_log,'.. Filtering with So parameters: big patch')
tic,
h=level;
fima2=onlm(ima,M,alpha+1,h);
fima2=fima2 - abs(mini);
t2(1)=toc;


% Hard wavelet Coefficient Mixing
j_disp(fname_log,'.. Hard wavelet Coefficient Mixing')
tic,
fima3 = hsm(fima1,fima2);
t3(1)=toc;
t3(1)=t3(1)+t2(1)+t1(1);


% % Adaptive wavelet coefficient Mixing
% j_disp(fname_log,'.. Adaptive wavelet coefficient Mixing')
% tic,
% fima4 = ascm(ima,fima1,fima2,level);
% t4(1)=toc;
% t4(1)=t4(1)+t2(1)+t1(1);


% % save filtered data
% j_disp(fname_log,['Save filtered data ONLM...'])
% fname_save = [fname_out,'_onlm'];
% j_disp(fname_log,['.. output name: ',fname_save])
% save_avw(fima1,fname_save,'f',scales(1:3));
% j_disp(fname_log,['\nCopy geometric information from ',fname_out,'...'])
% cmd = [fsloutput,'fslcpgeom ',fname_in,' ',fname_save,' -d'];
% j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end

% save filtered data
j_disp(fname_log,['Save filtered data ONLM-HSM...'])
fname_save = [fname_out];
j_disp(fname_log,['.. output name: ',fname_save])
save_avw_v2(fima3,fname_save,'f',scales(1:3));
j_disp(fname_log,['\nCopy geometric information from ',fname_out,'...'])
cmd = [fsloutput,'fslcpgeom ',fname_in,' ',fname_save,' -d'];
j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end

% % save filtered data
% j_disp(fname_log,['Save filtered data ONLM-ASCM...'])
% fname_save = [fname_out,'_onlm-ascm'];
% j_disp(fname_log,['.. output name: ',fname_save])
% save_avw(fima4,fname_save,'f',scales(1:3));
% j_disp(fname_log,['\nCopy geometric information from ',fname_out,'...'])
% cmd = [fsloutput,'fslcpgeom ',fname_in,' ',fname_save,' -d'];
% j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end


% END FUNCTION
j_disp(fname_log,['\n.. Ended: ',datestr(now)])
j_disp(fname_log,['==========================================================================================================\n'])
