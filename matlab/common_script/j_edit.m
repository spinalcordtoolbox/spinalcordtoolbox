function j_edit(varargin)
% TEDIT create a file using a template.
%   TEDIT(funname) opens the editor and pastes the content
%   of a user-defined template into the file funname.m.
% 
%   Example
%   tedit myfunfun
%   opens the editor and pastes the following 
% 
% 	function output = myfunfun(input)
% 	%MYFUNFUN  One-line description here, please.
% 	%   output = myfunfun(input)
% 	%
% 	%   Example
% 	%   myfunfun
% 	%
% 	%   See also
% 
% 	% Author: Julien Cohen-Adad <email>
% 	% Created: 2005-09-22
% 	% Copyright 2005 Your company.
% 
%   See also edit, mfiletemplate

% Author: Peter (PB) Bodin
% Created: 2005-09-22
	
	
	% See the variables repstr, repwithstr and tmpl to figure out how
	% to design your own template.
	% Edit tmpl to your liking, if you add more tokens in tmpl, make
	% sure to add them in repstr and repwithstr as well.
	
	% I made this function just for fun to check out some java handles to
	% the editor. It would probably be better to fprintf the template
	% to a new file and then call edit, since he java objects might change
	% names between versions.

	switch nargin
		case 0
			edit
			warning('tedit without argument is the same as edit')
			return;
		case 1
			fname=varargin{:};
			edit(fname);
		otherwise
			error('too many input arguments')
	end

	try
		edhandle=com.mathworks.mlservices.MLEditorServices;
		edhandle.builtinAppendDocumentText(strcat(fname,'.m'),parse(fname));
	catch
		rethrow(lasterr)
	end

	function out = parse(func)

		tmpl={ ...
			'function opt = $filename(opt)'
			'% ========================================================================='
			'% '
			'% '
			'% '
			'% INPUT'
			'% -------------------------------------------------------------------------'
			'% (opt)' 
			'%   fname_log			string'
			'% '
			'% -------------------------------------------------------------------------'
			'%'
			'% OUTPUT'
			'% -------------------------------------------------------------------------'
			'% '
			'% -------------------------------------------------------------------------'
			'% '
			'%   Example'
			'%   $filename'
			'%'
			'%'
			'% $author'
			'% $date: Created'
			'%'
			'% ========================================================================='
			''
			'% PARAMETERS'
			''
			''
			'% Check number of arguments'
			'if nargin < 2'
			'	disp(''Not enought arguments. Type: help $filename'')'
			'	return'
			'end'
			''
			'% INITIALIZATION'
			'dbstop if error; % debug if error'
			'if ~exist(''opt''), opt = []; end'
			'if isfield(opt,''fname_log''), fname_log = opt.fname_log, else fname_log = ''log_$filename.txt''; end'
			''
			''
			'% START FUNCTION'
			'j_disp(fname_log,[''\n\n\n==========================================================================================================''])'
			'j_disp(fname_log,[''   Running: $filename''])'
			'j_disp(fname_log,[''==========================================================================================================''])'
			'j_disp(fname_log,[''.. Started: '',datestr(now)])'
			''
			'% Check parameters'
			'j_disp(fname_log,[''\nCheck parameters:''])'
			''
			''
			'% Find which SHELL is running'
			'j_disp(fname_log,[''\nFind which SHELL is running...''])'
			'[status result] = unix(''echo $0'');'
			'if ~isempty(findstr(result,''bash''))'
			'	shell = ''bash'';'
			'elseif ~isempty(findstr(result,''tsh''))'
			'	shell = ''tsh'';'
			'elseif ~isempty(findstr(result,''tcsh''))'
			'	shell = ''tcsh'';'
			'else'
			'	j_disp(fname_log,[''.. Failed to identify shell. Using default.''])'
			'	shell = ''bash'';'
			'end'
			'j_disp(fname_log,[''.. Running: '',shell])'
			''
			''
			''
			'% FSL output'
			'if strcmp(shell,''bash'')'
			'	fsloutput = [''export FSLOUTPUTTYPE='',outputtype,''; '']; % if running BASH'
			'	fsloutput_temp = [''export FSLOUTPUTTYPE=NIFTI; ''];'
			'elseif strcmp(shell,''tsh'') || strcmp(shell,''tcsh'') '
			'	fsloutput = [''set FSLOUTPUTTYPE='',outputtype,''; '']; % if you''re running T-SHELL'
			'	fsloutput_temp = [''setenv FSLOUTPUTTYPE NIFTI; ''];'
			'end'
			''
			''
			''
			'% END FUNCTION'
			'j_disp(fname_log,[''\n.. Ended: '',datestr(now)])'
			'j_disp(fname_log,[''==========================================================================================================\n''])'
			};

		repstr={...
			'$filename'
			'$FILENAME'
			'$date'
			'$year'
			'$author'
			'$company'};

		repwithstr={...
			func
			upper(func)
			datestr(now,29)
			datestr(now,10)
			'Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>'
			'Your company'};

		for k = 1:numel(repstr)
			tmpl = strrep(tmpl,repstr{k},repwithstr{k});
		end
		out = sprintf('%s\n',tmpl{:});
	end
end