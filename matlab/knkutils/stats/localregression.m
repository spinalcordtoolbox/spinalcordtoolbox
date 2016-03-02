function f = localregression(x,y,x0,degree,kernel,h,mode)

% function f = localregression(x,y,x0,degree,kernel,h,mode)
%
% <x>,<y> are vectors with the data
% <x0> is a vector with the points to evaluate at
% <degree> (optional) is non-negative integer.  default: 1.
% <kernel> (optional) is 'epan'.  default: 'epan'.
% <h> (optional) is [A B C] where A is the bandwidth, B is the
%   minimum number of points that must exist in order to perform
%   a regression, and C is whether to automatically refuse to
%   perform a regression if an <x0> point lies outside the range of <x>.
%   A can be Inf.  if not specified, B defaults to 1 and C defaults
%   to 0.  default: [std(x)/10 1 0].
% <mode> (optional) is
%   0 means L2 and we use pinv to fit the various local linear models.
%   1 means L1 and we use fitl1line to fit the various local linear models.
%   default: 0.
%
% return a vector with the value of the function at <x0>.
%
% singular warnings are suppressed.  can return NaNs.
% note that entries with NaN in <x> or <y> are ignored.
%
% see also localregression3d.m.
%
% note that we use parfor as a way to potentially speed up execution.
% if parallelization is used, note that status dots are outputted only at the end.
%
% note that local regression is also known as LOWESS (locally 
% weighted scatterplot smoothing).
%
% example:
% x = randn(1,1000);
% y = sin(x) + .2*randn(size(x));
% x0 = -2:.1:2;
% figure; hold on;
% scatter(x,y,'r.');
% plot(x0,localregression(x,y,x0,[],[],.5),'b-','LineWidth',3);
% plot(x0,localregression(x,y,x0,[],[],2),'k-','LineWidth',3);

% TODO: SHOULD WE PULL "WTS" FUNCTIONALITY FROM LOCALREGRESSION3.M?
  % SHOULD WE PULL OTHER CHANGES FROM LOCALREGRESSION3D AND LOCALREGRESSION4D ?

% TODO: SHOULD WE EXTEND L1 CASE TO OTHER VERSIONS OF LOCALREGRESSION?

% TODO: SHOULD WE EXTEND H MINIMUM TO OTHER VERSIONS?

% input
if ~exist('degree','var') || isempty(degree)
  degree = 1;
end
if ~exist('kernel','var') || isempty(kernel)
  kernel = 'epan';
end
if ~exist('h','var') || isempty(h)
  h = std(x)/10;
end
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end
if length(h)==2
  h = [h 0];
end
if length(h)==1
  h = [h 1 0];
end

% deal with NaN
bad = isnan(x) | isnan(y);
x(bad) = [];
y(bad) = [];

% get out early
if isempty(x)
  f = NaN*ones(size(x0));
  return;
end

% calc
mn = min(x(:));
mx = max(x(:));

  prev = warning('query'); warning('off');

% do it
f = NaN*zeros(size(x0));
parfor pp=1:length(x0)
  
  % if auto-rejection desired, see if we need to do it
  if h(3) && (x0(pp) < mn || x0(pp) > mx)
    continue;
  end

  % figure out limited support and calculate kernel weights
  temp = abs(x-x0(pp))/h(1);
  good = temp <= 1;
  k = 0.75*(1-temp(good).^2);
  
  % filter out
  xA = x(good);
  yA = y(good);
  n = length(xA);
  
  % form X matrices
  X = []; x0X = [];
  for p=0:degree
    if p==0
      X = cat(2,X,ones(n,1));
      x0X = [x0X 1];
    elseif p==1
      X = cat(2,X,xA');
      x0X = [x0X x0(pp)];
    else
      X = cat(2,X,xA'.^p);
      x0X = [x0X x0(pp).^p];
    end
  end

% AVOID THIS FOR SPEED REASONS
%  % form W matrix
%  W = diag(k);
  
  % if we have enough data points, fit the model
  if size(X,1) >= size(X,2) && size(X,1) >= h(2)

    % solve it
      warning('off');  % hopefully this doesn't slow things down.  [IS THIS REALLY NECESSARY ANY MORE?]
  % OLD WAY: %  sol = (X'*bsxfun(@times,k',X)) \ (X'*(k'.*yA'));
    X = bsxfun(@times,sqrt(k'),X);
    yA = bsxfun(@times,sqrt(k'),yA');
    switch mode
    case 0
      sol = pinv(X)*yA;
    case 1
      sol = fitl1line(X,yA,[])';
    end
    assert(all(isfinite(sol)));
    f(pp) = x0X*sol;

  end

end

  warning(prev);
