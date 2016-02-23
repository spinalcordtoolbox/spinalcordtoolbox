function sct_dmri_OrderByBvals(fname_data,fname_orderingfile,opt)

% =========================================================================
% Module that reorganizes diffusion-weighted data by bvals (using bvals file or schemefile)
%
% sct_dmri_OrderByBvals(fname_data,fname_orderingfile,opt)
% 
% INPUT
% fname_data			string
% fname_orderingfile	string   bvals file or schemefile
% (opt)
%   output_suffix		string. Default='_ordered'
%   fname_log
%   outputtype			NIFTI | NIFTI_GZ
%   bash				'bash'* | 'tsh'
% 
% OUTPUT
% -
%
% Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2013-09-04: created
% 
% =========================================================================

dbstop if error
% INITIALIZATION
if nargin<2, help sct_dmri_OrderByBvals, return, end
% dbstop if error; % debug if error
if ~exist('opt'), opt = []; end
if isfield(opt,'output_suffix'), output_suffix = opt.output_suffix; else output_suffix = '_ordered'; end
if isfield(opt,'fname_log'), fname_log = opt.fname_log; else fname_log = 'log_sct_dmri_OrderByBvals.txt'; end
if isfield(opt,'outputtype'), outputtype = opt.outputtype; else outputtype = 'NIFTI'; end
if isfield(opt,'shell'), shell = opt.shell; else shell = 'bash'; end



% =========================================================================
% START THE SCRIPT
% =========================================================================



% START FUNCTION
j_disp(fname_log,['\n\n\n=========================================================================================================='])
j_disp(fname_log,['   Running: sct_dmri_OrderByBvals'])
j_disp(fname_log,['=========================================================================================================='])
j_disp(fname_log,['.. Started: ',datestr(now)])


fname_output = [sct_tool_remove_extension(fname_data,1),output_suffix];
% Check parameters
j_disp(fname_log,['\nCheck parameters:'])
j_disp(fname_log,['.. Input data:        ',fname_data])
j_disp(fname_log,['.. ordering file:     ',fname_orderingfile])
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



fname_data=sct_tool_remove_extension(fname_data,1);






j_disp(fname_log, 'Ordering data by bvalues... ' );
% Split T
cmd = [fsloutput 'fslsplit ' fname_data ' T -t'];
j_disp(fname_log, ['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
num = j_numbering(2000,4,0);

% Scheme file or bval vector??
j_disp(fname_log, 'schemefile or bvals file? Checking file extension...');
if ~isempty(strfind(fname_orderingfile,'.scheme'))
    j_disp(fname_log, 'file extension is ".scheme"');
    
    % Create index
    % load scheme files
    scheme=scd_schemefile_read(fname_orderingfile);
    bvals = (2*pi*scheme(:,8)).^2.*(scheme(:,5) - scheme(:,6)/3)*10^3; %s/mm^2
    
    
else
    j_disp(fname_log, 'file extension isn''t ".scheme", assuming a bvals file');
    j_disp(fname_log,['.. File bvals: ',fname_orderingfile])
    % Create index
    bvals = load(fname_orderingfile);
    
    
end

    % sort by bvals
    [dummyvar,index_bvals] = sort(bvals); index_bvals=index_bvals(:);
    j_disp(fname_log, ['bvalues index : ' num2str(index_bvals')]);
    
    cmd = [fsloutput 'fslmerge -t ' fname_output];
    
    for i_bval = 1:length(bvals)
        cmd = [cmd ' T' num{index_bvals(i_bval)}];
    end
    j_disp(fname_log, ['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
    
    % remove split files
    unix('rm T*.nii');

    
j_disp(fname_log,['\n.. Ended: ',datestr(now)])
j_disp(fname_log,['==========================================================================================================\n'])

end