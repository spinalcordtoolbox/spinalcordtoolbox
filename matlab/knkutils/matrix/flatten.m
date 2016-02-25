function f = flatten(m)

% function f = flatten(m)
%
% <m> is a matrix
% 
% return as a row vector.
%
% example:
% a = [1 2; 3 4];
% isequal(flatten(a),[1 3 2 4])

f = m(:).';
