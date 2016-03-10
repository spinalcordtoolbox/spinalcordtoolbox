function result = unix_wrapper(cmd,wantreport,wantassert)

% function result = unix_wrapper(cmd,wantreport,wantassert)
%
% <cmd> is a string
% <wantreport> (optional) is whether to report to command window.  default: 1.
% <wantassert> (optional) is whether to assert that status==0.  default: 1.
%
% report <cmd> to the command window and then call <cmd>.
% after <cmd> is finished, report the status and the result
%   to the command window.
% then, if <wantassert>, assert that the status returned is 0.
%   if the status returned is not 0, we always display the 
%   result to the command window.
% finally, return the result.

% input
if ~exist('wantreport','var') || isempty(wantreport)
  wantreport = 1;
end
if ~exist('wantassert','var') || isempty(wantassert)
  wantassert = 1;
end

% do it
if wantreport
  fprintf('calling unix command:\n%s\n',cmd);
end
[status,result] = unix(cmd);
if wantreport
  fprintf('status of unix command:\n%d\n',status);
  fprintf('result of unix command:\n%s\n',result);
end
if wantassert
  if status~=0
    fprintf('unix command failed.  here was the result: \n%s\n',result);
  end
  assert(status==0);
end




%fprintf(1,'result of unix command: \n');
%disp(result);  % for weird escape characters, etc.
%fprintf(1,'\n');
