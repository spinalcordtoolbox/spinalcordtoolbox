function f = vflatten(m)

% function f = vflatten(m)
%
% <m> is a matrix
% 
% return as a column vector.
%
% example:
% a = [1 2; 3 4];
% isequal(vflatten(a),[1 3 2 4]')

f = m(:);
