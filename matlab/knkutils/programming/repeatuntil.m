function f = repeatuntil(fun,condfun)

% function f = repeatuntil(fun,condfun)
% 
% <fun> is a function that returns a matrix
% <condfun> is a function that can be applied to the output of <fun>
%   and returns a logical value
%
% call <fun> until <condfun> is satisfied and return that output.
%
% example:
% repeatuntil(@() rand,@(x) x > .9)

while 1
  f = feval(fun);
  if feval(condfun,f)
    return;
  end
end
