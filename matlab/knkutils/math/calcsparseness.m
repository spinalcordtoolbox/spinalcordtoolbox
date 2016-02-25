function f = calcsparseness(m,dim)

% function f = calcsparseness(m,dim)
%
% <m> is a matrix
% <dim> (optional) is the dimension of interest.
%   if supplied, calculate sparseness of each case oriented along <dim>.
%   if [] or not supplied, calculate sparseness of entire matrix.
%
% calculate sparseness of <m>, either of individual cases (in which case
% the output is the same as <m> except collapsed along <dim>) or globally
% (in which case the output is a scalar).  for the definition of sparseness,
% see Vinje Science 2000.
%
% we ignore NaNs gracefully.
% 
% note some weird cases:
%   calcsparseness([]) is [].
%   calcsparseness([NaN NaN]) is NaN
%
% example:
% isequal(calcsparseness([0 0 0 0 1]),1)

% handle weird case up front
if isempty(m)
  f = [];
  return;
end

% do it
if ~exist('dim','var') || isempty(dim)
  m = m(:);
  dim = 1;
end
len = sum(~isnan(m),dim);
f = zerodiv(1 - zerodiv(zerodiv(nansum(m,dim),len,NaN).^2,zerodiv(nansum(m.^2,dim),len,NaN),NaN,0),1 - zerodiv(1,len,NaN),NaN);
