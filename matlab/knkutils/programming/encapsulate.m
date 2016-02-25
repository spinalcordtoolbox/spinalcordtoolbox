function f = encapsulate(fun,idx)

% function f = encapsulate(fun,idx)
% 
% <fun> is a function that accepts a scalar
% <idx> is a vector (of scalars)
% 
% return a cell vector that encapsulates the output of
% <fun> evaluated at each element of <idx>.
%
% example:
% isequal(encapsulate(@(x) x^2,1:3),{1 4 9})

f = {};
for p=1:length(idx)
  f{p} = feval(fun,idx(p));
end
