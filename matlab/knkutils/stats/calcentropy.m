function f = calcentropy(m,dim,wantnorm)

% function f = calcentropy(m,dim,wantnorm)
%
% <m> is a matrix with nonnegative integers indicating counts
% <dim> (optional) is the dimension of interest.
%   if supplied, calculate entropy for cases oriented along <dim>.
%   default is [] which means to calculate entropy globally.
% <wantnorm> (optional) is whether to normalize with respect to the maximum possible entropy,
%   i.e. -log2(1/n) where n is the number of bins.  if so, the output will range between
%   0 and 1.  default: 0.  (a special case is n==1, in which case we will return NaN.)
%
% return the log2-entropy of <m>, operating either on individual cases or globally.
% in the former case, the result has the same dimensions as <m> except 
% collapsed along <dim>.  in the latter case, the result is a scalar.
%
% if all of the elements in a case are 0, then we return NaN for that case.
% if one of the elements in a case is NaN, then we return NaN for that case.
% if <m> is empty, we return [].
%
% example:
% calcentropy([1 1 1],[],1)

% input
if ~exist('dim','var') || isempty(dim)
  m = m(:);
  dim = 1;
end
if ~exist('wantnorm','var') || isempty(wantnorm)
  wantnorm = 0;
end

% handle weird case
if isempty(m)
  f = [];
  return;
end

% do it
numbins = size(m,dim);
m = bsxfun(@(x,y) zerodiv(x,y,NaN),m,sum(m,dim));
m(m==0) = 1;  % any element that is 0 is hacked so that it will not contribute anything to the summation
f = -sum(m.*log2(m),dim);

% normalize
if wantnorm
  f = zerodiv(f,-log2(1/numbins),NaN);
end
