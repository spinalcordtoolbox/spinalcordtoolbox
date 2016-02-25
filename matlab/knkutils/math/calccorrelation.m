function f = calccorrelation(x,y,dim,wantmeansubtract,wantgainsensitive)

% function f = calccorrelation(x,y,dim,wantmeansubtract,wantgainsensitive)
%
% <x>,<y> are matrices with the same dimensions
% <dim> (optional) is the dimension of interest.
%   default to 2 if <x> is a (horizontal) vector and to 1 if not.
%   special case is 0 which means to calculate globally.
% <wantmeansubtract> (optional) is whether to subtract mean first. default: 1.
% <wantgainsensitive> (optional) is whether to make the metric sensitive
%   to gain. default: 0.
%
% calculate Pearson's correlation coefficient between <x> and <y> for
% each case oriented along <dim>.  this is achieved by taking
% the two vectors implicated in each case, subtracting each vector's mean,
% normalizing each vector to have unit length, and then taking their dot product.
%
% a special case is when <wantmeansubtract>==0, in which case we omit the mean-
% subtraction.  this technically isn't Pearson's correlation any more.
%
% another special case is <wantgainsensitive>, in which case we take the two
% vectors implicated in each case, subtract each vector's mean (if 
% <wantmeansubtract>), normalize the two vectors by the same constant
% such that the larger of the two vectors has length 1, and then take the dot-
% product.  note that this metric still ranges from -1 to 1 and that any gain
% mismatch can only hurt you (i.e. push you towards a correlation of 0).
% (in retrospect, we now observe that it is probably better to use calccod.m in order
% to be sensitive to gain, rather than to use <wantgainsensitive> in this function.)
%
% if there is no variance in one (or more) of the inputs, the 
% result is returned as NaN.
%
% NaNs are handled gracefully (a NaN causes that data point to be ignored).
%
% if there are no valid data points (i.e. all data points are
% ignored because of NaNs), we return NaN for that case.
%
% we don't use caution with respect to cases involving low variance (see unitlength.m).
%
% be careful of the presumption that the mean and scale of <x> and <y> can be discounted.
% if you do not want to perform this discounting, use calccod.m instead!
%
% note some weird cases:
%   calccorrelation([],[]) is []
%
% example:
% calccorrelation(randn(1,100),randn(1,100))

% input
if ~exist('dim','var') || isempty(dim)
  if isvector(x) && size(x,1)==1
    dim = 2;
  else
    dim = 1;
  end
end
if ~exist('wantmeansubtract','var') || isempty(wantmeansubtract)
  wantmeansubtract = 1;
end
if ~exist('wantgainsensitive','var') || isempty(wantgainsensitive)
  wantgainsensitive = 0;
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

% propagate NaNs (i.e. ignore invalid data points)
x(isnan(y)) = NaN;
y(isnan(x)) = NaN;

% subtract mean
if wantmeansubtract
  x = zeromean(x,dim);
  y = zeromean(y,dim);
end

% normalize x and y to be unit length
[xnorm,xlen] = unitlength(x,dim,[],0);
[ynorm,ylen] = unitlength(y,dim,[],0);

% if gain sensitive, then do something tricky:
if wantgainsensitive

  % case where x is the bigger vector
  temp = xnorm .* bsxfun(@(x,y) zerodiv(x,y,NaN,0),y,xlen);
  bad = all(isnan(temp),dim);
  f1 = nansum(temp,dim);
  f1(bad) = NaN;

  % case where y is the bigger vector
  temp = bsxfun(@(x,y) zerodiv(x,y,NaN,0),x,ylen) .* ynorm;
  bad = all(isnan(temp),dim);
  f2 = nansum(temp,dim);
  f2(bad) = NaN;
  
  % figure out which one to use
  f = f1;                              % this is when x is bigger
  f(xlen < ylen) = f2(xlen < ylen);    % this is when y is bigger

% if not, just take the dot product
else
  temp = xnorm .* ynorm;
    % at this point, we want to sum along dim.  however, if a case has all NaNs, we need the output to be NaN.
  bad = all(isnan(temp),dim);  % record bad cases
  f = nansum(temp,dim);        % do the summation
  f(bad) = NaN;                % explicitly set bad cases to NaN
end
