function m = divnorm(m,n,sigma)

% function m = divnorm(m,n,sigma)
%
% <m> is a matrix
% <n> is an exponent
% <sigma> is a constant.  special case is Inf which
%   means to omit the denominator.
%
% compute m.^n ./ (m.^n + sigma.^n).
%
% note that there is some overlap with evaldivnorm.m.
%
% in general, we assume that <m>, <n>, and <sigma> are
% all non-negative and finite (except for the special
% case where <sigma> is Inf).
%
% we do some special handling for crazy cases.
% specifically, in the case that <sigma> is not Inf,
% we check whether m.^n is Inf.  if it is, then without
% special handling, the output would turn out to be NaN,
% which is bad.  so instead, we do the following:
%   (1) if m > sigma, then the output is set to 1.
%   (2) if m == sigma, then the output is set to 1/2.
%   (3) if m < sigma, then the output is set to 0.
% however, we do not handle the crazy case where
% both m.^n and sigma.^n are 0.  this case will
% result in NaN.
%
% example:
% isequal(divnorm(2,2,2),0.5)

if n==1 && isinf(sigma)
  return;
end

morig = m;  % save a copy
if n ~= 1  % speed
  m = m.^n;
end
if sigma==Inf
  % do nothing
else
  bad = isinf(m);
  bad2 = morig > sigma;
  bad3 = morig == sigma;
  m = m ./ (m + sigma.^n);
  m(bad & bad2) = 1;
  m(bad & ~bad2 & bad3) = 1/2;
  m(bad & ~bad2 & ~bad3) = 0;
end
