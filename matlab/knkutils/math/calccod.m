function f = calccod(x,y,dim,wantgain,wantmeansub)

% function f = calccod(x,y,dim,wantgain,wantmeansub)
%
% <x>,<y> are matrices with the same dimensions
% <dim> (optional) is the dimension of interest.
%   default to 2 if <x> is a (horizontal) vector and to 1 if not.
%   special case is 0 which means to calculate globally.
% <wantgain> (optional) is
%   0 means normal
%   1 means allow a gain to be applied to each case of <x>
%     to minimize the squared error with respect to <y>.
%     in this case, there cannot be any NaNs in <x> or <y>.
%   2 is like 1 except that gains are restricted to be non-negative.
%     so, if the gain that minimizes the squared error is negative,
%     we simply set the gain to be applied to be 0.
%   default: 0.
% <wantmeansub> (optional) is
%   0 means do not subtract any mean.  this makes it such that
%     the variance quantification is relative to 0.
%   1 means subtract the mean of each case of <y> from both
%     <x> and <y> before performing the calculation.  this makes
%     it such that the variance quantification
%     is relative to the mean of each case of <y>.
%     note that <wantgain> occurs before <wantmeansub>.
%   default: 1.
%
% calculate the coefficient of determination (R^2) indicating
% the percent variance in <y> that is explained by <x>.  this is achieved
% by calculating 100*(1 - sum((y-x).^2) / sum(y.^2)).  note that
% by default, we subtract the mean of each case of <y> from both <x>
% and <y> before proceeding with the calculation.
% 
% the quantity is at most 100 but can be 0 or negative (unbounded).  
% note that this metric is sensitive to DC and scale and is not symmetric 
% (i.e. if you swap <x> and <y>, you may obtain different results).  
% it is therefore fundamentally different than Pearson's correlation 
% coefficient (see calccorrelation.m).
%
% NaNs are handled gracefully (a NaN causes that data point to be ignored).
%
% if there are no valid data points (i.e. all data points are
% ignored because of NaNs), we return NaN for that case.
%
% note some weird cases:
%   calccod([],[]) is []
%
% history:
% 2013/08/18 - fix pernicious case where <x> is all zeros and <wantgain> is 1 or 2.
% 2010/11/28 - add <wantgain>==2 case
% 2010/11/23 - changed the output range to percentages.  thus, the range is (-Inf,100].
%              also, we removed the <wantr> input since it was dumb.
%
% example:
% x = randn(1,100);
% calccod(x,x+0.1*randn(1,100))

% input
if ~exist('dim','var') || isempty(dim)
  if isvector(x) && size(x,1)==1
    dim = 2;
  else
    dim = 1;
  end
end
if ~exist('wantgain','var') || isempty(wantgain)
  wantgain = 0;
end
if ~exist('wantmeansub','var') || isempty(wantmeansub)
  wantmeansub = 1;
end

% handle weird case up front
if isempty(x)
  f = [];
  return;
end

% handle 0 case
if dim==0
  x = x(:);
  y = y(:);
  dim = 1;
end

% handle gain
if wantgain
  % to get the residuals, we want to do something like y-x*inv(x'*x)*x'*y
  temp = 1./dot(x,x,dim) .* dot(x,y,dim);
  if wantgain==2
    temp(temp < 0) = 0;  % if the gain was going to be negative, rectify it to 0.
  end
  x = bsxfun(@times,x,temp);
end

% propagate NaNs (i.e. ignore invalid data points)
x(isnan(y)) = NaN;
y(isnan(x)) = NaN;

% handle mean subtraction
if wantmeansub
  mn = nanmean(y,dim);
  y = bsxfun(@minus,y,mn);
  x = bsxfun(@minus,x,mn);
end

% finally, compute it
f = 100*(1 - zerodiv(nansum((y-x).^2,dim),nansum(y.^2,dim),NaN,0));






% JUNK:
%
% % <wantr> (optional) is whether to apply signedarraypower(f,0.5)
% %   at the very end, giving something like a correlation coefficient (r).
% %   default: 1.
% 
% if ~exist('wantr','var') || isempty(wantr)
%   wantr = 1;
% end
% 
% if wantr
%   f = signedarraypower(f,0.5);
% end
