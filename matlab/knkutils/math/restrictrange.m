function f = restrictrange(m,mn,mx)

% function f = restrictrange(m,mn,mx)
%
% <m> is a matrix
% <mn> is a value
% <mx> is a value
%
% truncate the values of <m> to fit in the range [<mn>,<mx>].
%
% example:
% isequal(restrictrange([1 2 3 4],2.5,3),[2.5 2.5 3 3])

f = m;
f(f<mn) = mn;
f(f>mx) = mx;
