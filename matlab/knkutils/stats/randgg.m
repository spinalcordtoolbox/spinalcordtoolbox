function [f,table] = randgg(params,dims,table)

% function [f,table] = randgg(params,dims,table)
%
% <params> is [mn alpha beta] where
%   mn is the mean
%   alpha is the scaling (positive)
%   beta is the exponent (positive)
% <dims> is the matrix dimension you want
% <table> (optional) is a speed-up (dependent on <params>)
%
% generate random numbers from the generalized Gaussian distribution
% specified by <params>.  note that the result is (slightly) approximate,
%   since we approximate the CDF using piecewise linear interpolation.
%
% example:
% [a,table] = randgg([0 1 1],[1 10000]);
% figure; hist(a,1000);
% [params,d,d,exitflag] = lsqnonlin(@(x) sqrt(-log(evalgeneralizedgaussian(x,a))),[1 2 2],[-Inf 0 0],[Inf Inf Inf],optimset('Display','iter','MaxIter',Inf));
% params

% NOTE: see randggd.m??

% constant
thresh = 1e-11;  % we assume we will never encounter a rand value within this much distance from 0 or 1

% do we need to calculate the table?
if ~exist('table','var') || isempty(table)

  % constants
  numsd = 20;  % above and below this many sd
  numpointspersd = 100;  % points within one sd chunk
  
  % input
  mn = params(1);
  alpha = params(2);
  beta = params(3);
  
  % calc std dev
  sd = sqrt(alpha^2*gamma(3/beta) / gamma(1/beta));

  % get the x-values
  xx = linspace(mn-numsd*sd,mn+numsd*sd,2*numsd*numpointspersd+1);
  
  % evaluate the cdf
  yy = 1/2 + sign(xx-mn)/2 .* gammainc((abs(xx-mn)/alpha).^beta,1/beta);
  
  % make sane (interp1 gets mad when the x-values are too close together)
  bad = yy < thresh | yy > 1-thresh;
  xx = xx(~bad);
  yy = yy(~bad);
  
  % save
  table = {xx yy};

end

% do it
f = interp1(table{2},table{1},rand(dims),'linear',NaN);
assert(all(~isnan(f(:))),'oops, our table was not big enough. need to re-think our code.');
