function m = signforce(m)

% function m = signforce(m)
%
% <m> is a matrix
%
% positive or zero elements are set to 1.
% negative elements are set to -1.
%
% note that this routine is like sign.m except that we don't allow 0 cases.
%
% example:
% isequal(signforce([2 0 -4]),[1 1 -1])

check = m>=0;
m(check) = 1;
m(~check) = -1;
