function f = allzero(m,thresh)

% function f = allzero(m,thresh)
%
% <m> is a matrix
% <thresh> (optional) is the threshold.  default: 1e-5.
%
% return whether all elements of <m> have an absolute value less than <thresh>.
%
% example:
% allzero([eps eps])

% inputs
if ~exist('thresh','var') || isempty(thresh)
  thresh = 1e-5;
end

% do it
f = all(abs(m(:)) < thresh);
