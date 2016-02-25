function f = discretize(m,x,y)

% function f = discretize(m,x,y)
%
% <m> is a matrix
% <x>,<y> are two numbers
%
% for each element of <m>, replace it with either <x> or <y> depending on
% which one is closer.  if exactly in-between, round up.
%
% example:
% discretize([0 .3 .5 .9],0,1)

ok = sort([x y]);
f = repmat(ok(1),size(m));
f(m >= (x+y)/2) = ok(2);
