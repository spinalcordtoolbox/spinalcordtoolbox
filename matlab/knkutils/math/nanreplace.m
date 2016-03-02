function m = nanreplace(m,val,mode)

% function m = nanreplace(m,val,mode)
%
% <m> is a matrix
% <val> (optional) is a scalar.  default: 0.
% <mode> (optional) is
%   0 means replace all NaNs in <m> with <val>.
%   1 means if the first element of <m> is not finite (i.e. NaN, -Inf, Inf), fill entire matrix with <val>.
%   2 means if it is not true that all elements of <m> are finite and real, fill entire matrix with <val>.
%   3 means replace any non-finite value in <m> in <val>.
%   default: 0.
%
% example:
% isequal(nanreplace([1 NaN],0),[1 0])
% isequal(nanreplace([NaN 2 3],0,1),[0 0 0])

% input
if ~exist('val','var') || isempty(val)
  val = 0;
end
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end

% do it
switch mode
case 0
  m(isnan(m)) = val;
case 1
  if ~isfinite(m(1))
    m(:) = val;
  end
case 2
  if ~all(isreal(m(:)) & isfinite(m(:)))
    m(:) = val;
  end
case 3
  m(~isfinite(m)) = val;
end
