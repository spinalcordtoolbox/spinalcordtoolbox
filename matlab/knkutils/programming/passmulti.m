function f = passmulti(fun,args,forceempty)

% function f = passmulti(fun,args,forceempty)
%
% <fun> is a function
% <args> is a cell vector of arguments to pass to <fun>
% <forceempty> (optional) is whether to pass [] to <fun>
%   when isempty(<args>).  the point is to ensure that <fun>
%   gets at least one argument.  default: 0.
%
% return something like fun(args{:}).  this function is useful for
% expanding out cell vectors on-the-fly, given that MATLAB doesn't 
% provide an easy way to apply "{:}" to an arbitrary matrix.
% 
% example:
% passmulti(@blkdiag,{randn(2,2) ones(2,2)})

% input
if ~exist('forceempty','var') || isempty(forceempty)
  forceempty = 0;
end

% do it
if isempty(args) && forceempty
  args = {[]};
end
f = feval(fun,args{:});
