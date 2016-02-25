function m = posrect(m)

% function m = posrect(m)
%
% <m> is a matrix
%
% positively-rectify <m>.
%
% example:
% isequal(posrect([2 3 -4]),[2 3 0])

m(m<0) = 0;
