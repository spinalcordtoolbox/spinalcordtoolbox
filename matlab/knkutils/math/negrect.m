function m = negrect(m)

% function m = negrect(m)
%
% <m> is a matrix
%
% negatively-rectify <m>.
%
% example:
% isequal(negrect([2 3 -4]),[0 0 -4])

m(m>0) = 0;
