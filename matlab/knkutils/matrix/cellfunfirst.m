function f = cellfunfirst(fun,m)

% function f = cellfunfirst(fun,m)
%
% <fun> is a function
% <m> is a cell matrix or a matrix
%
% if <m> is a cell matrix, apply <fun> to the first element of <m>.
% if <m> is not a cell matrix, apply <fun> to <m>.
%
% this function is useful in cases where you are not sure whether <m>
% has been encapsulated inside of a cell matrix.
%
% example:
% isequal(cellfunfirst(@(x) size(x,2),{[1 2 3] [3 4]}),3)
% isequal(cellfunfirst(@(x) size(x,2),[1 2 3]),3)

if iscell(m)
  f = feval(fun,m{1});
else
  f = feval(fun,m);
end
