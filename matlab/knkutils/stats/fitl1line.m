function [params,R2] = fitl1line(X,y,params0)

% function [params,R2] = fitl1line(X,y,params0)
%
% <X> is samples x parameters
% <y> is samples x 1
% <params0> (optional) is an initial seed.
%   default is [] which means use the OLS solution.
%
% use lsqnonlin.m to estimate the best linear weights that predict <y> from <X>.
% here, we define "best" in terms of the L1-norm of the residuals; that is,
% we minimize the sum of the absolute values of the residuals of the model fit
% (i.e. minimize the sum of the absolute deviations).  this is analogous to
% finding the median (instead of the mean) of the <y> given the <X>, and is
% thus appropriate for situations involving outliers, asymmetric distributions, etc.
%
% return:
%  <params> is 1 x parameters with the weights
%  <R2> is the R^2 between fitted and actual values (see calccod.m).
%
% example:
% x = sort(randn(1,100));
% y = x + abs(randn(1,100)).^2;
% X = [x' ones(100,1)];
% h = fitl1line(X,y');
% h2 = pinv(X)*y';
% figure; hold on;
% scatter(x,y,'r.');
% h1 = plot(x,X*h','b-');
% h2 = plot(x,X*h2,'g-');
% legend([h1 h2],{'L1' 'L2'});

% input
if ~exist('params0','var') || isempty(params0)
  params0 = [];
end

% ensure double to avoid problems
X = double(X);
y = double(y);
params0 = double(params0);

% define options
options = optimset('Display','off','FunValCheck','on','MaxFunEvals',Inf,'MaxIter',Inf,'TolFun',1e-6,'TolX',1e-6);

% define seed
if isempty(params0)
  params0 = (pinv(X)*y)';
end

% define bounds
paramslb = [];
paramsub = [];

% do it
[params,d,d,exitflag,output] = lsqnonlin(@(pp) sqrt(abs(y-X*pp')),params0,paramslb,paramsub,options);
assert(exitflag > 0);

% how well did we do?
R2 = calccod(X*params',y);
