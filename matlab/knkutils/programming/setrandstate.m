function setrandstate(flag)

% function setrandstate(flag)
%
% <flag> (optional) is
%   0 means seed to 0
%   1 means seed to sum(100*clock)
%   {N} means seed to N
%   default: 1.
%
% induce randomness by setting the state of rand and randn.
%
% example:
% setrandstate;

if ~exist('flag','var') || isempty(flag)
  flag = 1;
end
if iscell(flag)
  seed0 = flag{1};
elseif flag==0
  seed0 = 0;
else
  seed0 = sum(100*clock);
end
rand('state',seed0);
randn('state',seed0);
