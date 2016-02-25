function nr = PsychNumel(x)
% PTB-Replacement for numel() on older Matlab versions that don't support it.
%
% PsychNumel is a drop-in replacement for the numel() function of recent
% versions of Matlab. If called on modern Matlabs, it will just call
% numel(). On older Matlabs it will emulate the behaviour of numel().
%
% n = PsychNumel(x); will return the total number of elements contained in
% scalar, vector or matrix x, i.e. n == prod(size(x));
%

% History:
% 02/14/09  mk Written.

if exist('numel', 'builtin')
  % Call builtin implementation:
  nr = builtin('numel', x);
else
  % Use our fallback-implementation:
  if nargin < 1
      error('Not enough input arguments.');
  else
      nr = prod(size(x)); %#ok<PSIZE>
  end
end

return;
