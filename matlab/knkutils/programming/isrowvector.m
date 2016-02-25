function f = isrowvector(m)

% function f = isrowvector(m)
%
% <m> is a matrix
%
% return whether <m> is 1 x n where n >= 0.
% specifically:
%   f = isvector(m) & size(m,1)==1;
%
% example:
% isrowvector([1 2])
% isrowvector([1])
% isrowvector(zeros(1,0))
% ~isrowvector([])

f = isvector(m) & size(m,1)==1;
