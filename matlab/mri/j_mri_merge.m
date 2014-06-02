function j_mri_merge(param)
% =========================================================================
% 
% Merge data using FSL. This function was created because when the number of files is too large (>1000 aproximately), then the
% fslmerge function doesn't work. So this latbal matlab function slip the merging process.
% 
% 
% INPUT
% -------------------------------------------------------------------------
% param					structure.
% 
% *** MANDATORY ***
% 
%	fname_split			cell of string.  Data to merge.
%	fname_merge			string		output file name
% 
% *** OPTIONAL ***
% 
%	(fname_log)			string		fname to log file
% -------------------------------------------------------------------------
% 
% OUTPUT
% -------------------------------------------------------------------------
% -
% -------------------------------------------------------------------------
%
% Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2012-10-09: Created
% =========================================================================


% debug if error
dbstop if error

% Check number of arguments
if nargin < 1
	help j_mri_merge
	return
end


% INITIALIZATION
if isfield(param,'fname_split'), fname_split = param.fname_split; else error('fname_split field needed.'); end
if isfield(param,'fname_merge'), fname_merge = param.fname_merge; else error('fname_merge field needed.'); end
if isfield(param,'fname_log'), fname_log = param.fname_log; else fname_log = 'j_mri_merge.txt'; end





% =========================================================================
% START THE SCRIPT
% =========================================================================

% START FUNCTION
j_disp(fname_log,['\n\n\n=========================================================================================================='])
j_disp(fname_log,['   Running: j_mri_merge.m'])
j_disp(fname_log,['=========================================================================================================='])
j_disp(fname_log,['.. Started: ',datestr(now)])


% Check parameters
% j_disp(fname_log,['\nCheck parameters:'])
j_disp(fname_log,['.. data to merge:     ',fname_split{1}])
j_disp(fname_log,['.. data output:       ',fname_merge])
j_disp(fname_log,['.. fname log:         ',fname_log])


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



nt = size(param.fname_split,2);

% check if nt is big (might create problems, so in that case split the merging process)
if (nt < 1000)
	% nt is not big
	cmd = [fsloutput,'fslmerge -t ',fname_merge];
	for iT=1:nt
		cmd = cat(2,cmd,' ',fname_split{iT});
	end
	j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	
else

	% nt is big
	iTsub = 1;
	cmd_sub = [fsloutput,'fslmerge -t ',fname_merge,'_sub',num2str(iTsub)];
	iTbig = 1;
	for iT=1:nt
		cmd_sub = cat(2,cmd_sub,' ',fname_split{iT});
		if iTbig == 1000
			% output the subparts of the merge
			j_disp(fname_log,['>> ',cmd_sub]); [status result] = unix(cmd_sub); if status, error(result); end
			% increment
			iTsub = iTsub + 1;
			% reinitialize
			cmd_sub = [fsloutput,'fslmerge -t ',fname_merge,'_sub',num2str(iTsub)];
			iTbig = 1;
		end
		iTbig = iTbig + 1;
	end
	% output the last subpart of the merge
	j_disp(fname_log,['>> ',cmd_sub]); [status result] = unix(cmd_sub); if status, error(result); end
	
	% merge the subparts of the merge
	cmd = [fsloutput,'fslmerge -t ',fname_merge];		
	for iTbig=1:iTsub
		cmd = cat(2,cmd,' ',fname_merge,'_sub',num2str(iTbig));
	end
	j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end

	% remove temp files
	for iTbig=1:iTsub
		cmd = ['rm ',fname_merge,'_sub',num2str(iTbig),'.*'];
		j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	end
	
end

% end
j_disp(fname_log,['\n.. Ended: ',datestr(now)])
j_disp(fname_log,['==========================================================================================================\n'])

