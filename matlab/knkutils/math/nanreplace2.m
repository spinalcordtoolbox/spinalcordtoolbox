function m = nanreplace2(m,dim,val)

% function m = nanreplace2(m,dim,val)
%
% <m> is a matrix
% <dim> is a dimension of <m>
% <val> (optional) is a scalar.  default: 0.
%
% along dimension <dim>, if any element is not finite,
% fill that row with <val>.
%
% example:
% nanreplace2([1 2 NaN; 3 4 5],2)

% input
if ~exist('val','var') || isempty(val)
  val = 0;
end

% do it
m(fillout(any(~isfinite(m),dim),size(m))) = val;
