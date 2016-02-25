function f = gethostname

% function f = gethostname
%
% return the hostname (whitespace-stripped).
%
% example:
% gethostname

[status,result] = unix('hostname');
assert(status==0);
tokens = regexp(result,'(\S+?)($|\s)','tokens');
f = tokens{1}{1};
