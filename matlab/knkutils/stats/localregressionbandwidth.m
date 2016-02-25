function [hbest,err] = localregressionbandwidth(x,y,h,x0,fold,degree,kernel,mode)

% function [hbest,err] = localregressionbandwidth(x,y,h,x0,fold,degree,kernel,mode)
%
% <x>,<y> are vectors with the data
% <h> is a vector of bandwidths to try
% <x0> (optional) is a vector of nice x's to evaluate at and linearly interpolate in-between.
%   this vector should have a range that covers all points and should be at least as fine
%   as the smallest bandwidth.  if [] or not supplied, evaluate exactly at the held-out points.
%   the only point of <x0> is to increase execution speed.
% <fold> (optional) is positive integer > 1 with number of folds to do.  default: 10.
%   special case is -n which means act as if n but stop after first iteration.
% <degree> (optional) is non-negative integer.  default: 1.
% <kernel> (optional) is 'epan'.  default: 'epan'.
% <mode> (optional) is
%   0 means L2 and we use pinv to fit the various local linear models.
%   1 means L1 and we use fitl1line to fit the various local linear models.
%   default: 0.
%
% return:
%   <hbest> as the best bandwidth
%   <err> as a vector with average squared error averaged over folds
%
% note that the fold-resampling is deterministic!
% note that entries with NaN in <x> or <y> are ignored.
%
% example:
% x = randn(1,1000);
% y = sin(x) + .2*randn(size(x));
% [hbest,err] = localregressionbandwidth(x,y,.1:.1:1);
% figure; plot(err);
% x0 = -2:.1:2;
% figure; hold on;
% scatter(x,y,'r.');
% plot(x0,localregression(x,y,x0,[],[],hbest),'b-','LineWidth',3);
% title(sprintf('bandwidth = %.4f',hbest));

% input
if ~exist('x0','var') || isempty(x0)
  x0 = [];
end
if ~exist('fold','var') || isempty(fold)
  fold = 10;
end
if ~exist('degree','var') || isempty(degree)
  degree = 1;
end
if ~exist('kernel','var') || isempty(kernel)
  kernel = 'epan';
end
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end

% deal with NaN
bad = isnan(x) | isnan(y);
x(bad) = [];
y(bad) = [];

% calc
n = length(x);
if fold < 0
  numfolds = 1;
else
  numfolds = fold;
end

  prev = warning('query'); warning('off');

% do it
err = zeros(length(h),numfolds);
for p=1:length(h)
  fprintf('localregressionbandwidth: performing %d of %d\n',p,length(h));
  for q=1:numfolds
    holdout = picksubset(1:n,[abs(fold) q]);
    keepin = setdiff(1:n,holdout);
    
    if isempty(x0)
      vals = localregression(x(keepin),y(keepin),x(holdout),degree,kernel,h(p),mode);
    else
      vals = localregression(x(keepin),y(keepin),x0,degree,kernel,h(p),mode);
      vals = interp1(x0,vals,x(holdout),'linear');
    end
    err(p,q) = nanmean((vals-y(holdout)).^2);
  end
end
err = nanmean(err,2)';
[mn,ix] = min(err);
hbest = h(ix);

  warning(prev);
