function f = count(m)

% function f = count(m)
%
% <m> is a matrix
% 
% return the sum of all elements of <m>.
%
% if <m> is a logical matrix or a matrix of
% non-negative integers, this function can be naturally
% interpreted as counting the total number of things.
%
% example:
% count([1 1 0 1])==3

f = sum(m(:));
