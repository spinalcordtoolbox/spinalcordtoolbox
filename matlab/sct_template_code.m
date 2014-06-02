function sct_template_code(var1,opt)
% =========================================================================
% 
% This is just a template to show how the code should be structured (with of course some level of flexibility!)
% 
% 
% INPUT
% -------
% var1					string. explicit description.							<-- always put the type of the variable. var1 is mandatory
% (opt)																					<-- If an argument is optional, put it in (brackets).
%   var2					binary. 0 | 1*	 explicit description.				<-- the default value should be indicated.
%   fname_log			string.  log for processing.
% 
% OUTPUT
% -------
% (-)
%
% EXAMPLE
% -------
%	sct_template_code('my_data');
% 
% Author name <email address>
% Date of creation
% =========================================================================


% initialization
if nargin<1, help sct_template_code, return, end
if ~exist('opt'), opt = []; end
if isfield(opt,'var2'), var2 = opt.var2, else var2 = 1; end
if isfield(opt,'fname_log'), fname_log = opt.fname_log, else fname_log = 'log_sct_template_code.txt'; end


j_disp(fname_log,['\n\n\n=========================================================================================================='])
j_disp(fname_log,['   Running: sct_template_code.m'])
j_disp(fname_log,['=========================================================================================================='])
j_disp(fname_log,['.. Started: ',datestr(now)])

% Check parameters
j_disp(fname_log,['\nCheck parameters:'])
j_disp(fname_log,['.. Input data:            ',var1])
j_disp(fname_log,['.. bvecs file:            ',num2str(var2)])


% *** CODE HERE ***


% end
j_disp(fname_log,['\n.. Ended: ',datestr(now)])
j_disp(fname_log,['==========================================================================================================\n'])
