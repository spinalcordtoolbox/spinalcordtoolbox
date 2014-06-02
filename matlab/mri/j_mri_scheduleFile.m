function param = j_mri_scheduleFile(dof)
% =========================================================================
% 
% Generate schedule file to use with FLIRT
% 
% 
% INPUT
% -------------------------------------------------------------------------
% dof			'Ty_Sy_Kxy'* | 'Ty_Sy'.  Degree of freedom Ty = Translation along Y, Sy = scaling along Y, Kxy = shearing in X-Y plane. 
% 
% -------------------------------------------------------------------------
% OUTPUT
% -------------------------------------------------------------------------
% 
% -------------------------------------------------------------------------
% 
%   Example
%   j_mri_scheduleFile
%
%
% Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2011-11-19: Created
%
% =========================================================================

% PARAMETERS


% Check number of arguments
if nargin < 1
	disp('Not enought arguments. Type: help j_mri_scheduleFile')
	return(-1)
end

% INITIALIZATION
if ~exist('opt'), opt = []; end
if isfield(opt,'fname_log'), fname_log = opt.fname_log, else fname_log = 'log_j_mri_scheduleFile.txt'; end


% % START FUNCTION
% j_disp(fname_log,['\n\n\n=========================================================================================================='])
% j_disp(fname_log,['   Running: j_mri_scheduleFile'])
% j_disp(fname_log,['=========================================================================================================='])
% j_disp(fname_log,['.. Started: ',datestr(now)])

% Check parameters
% j_disp(fname_log,['\nCheck parameters:'])

fid = fopen(dof,'w');
% txt = schedule_Ty_Sy_Kxy();
txt = 
fclose(fid)

% % END FUNCTION
% j_disp(opt.fname_log,['\n.. Ended: ',datestr(now)])
% j_disp(opt.fname_log,['==========================================================================================================\n'])







function txt = schedule_Ty_Sy_Kxy()

