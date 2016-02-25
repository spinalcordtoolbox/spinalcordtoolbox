function m = negreplace(m,val)

% function m = negreplace(m,val)
%
% <m> is a matrix
% <val> is a scalar
% 
% replace all negative values in <m> with <val>.
%
% example:
% isequal(negreplace([1 -1],0),[1 0])

m(m < 0) = val;
