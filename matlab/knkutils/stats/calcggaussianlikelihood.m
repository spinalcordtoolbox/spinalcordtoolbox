function f = calcggaussianlikelihood(params,data)

% function f = calcggaussianlikelihood(params,data)
%
% <params> is [mn alpha beta] (see evalgeneralizedgaussian.m)
% <data> is points x dimensions with points to evaluate the likelihood at
%
% calculate the likelihood of a set of points given an assumption that the points
% are independently drawn from a generalized Gaussian distribution.
% return the average log probability value.
%
% example:
% calcggaussianlikelihood([0 1 2],randn(1000,50))

% calc
mn = params(1);
alpha = params(2);
beta = params(3);
numpoints = size(data,1);
numdims = size(data,2);

% do it
bb = sum(sum(abs(data-mn).^beta / (-alpha^beta),2),1);
f = numdims * log2(beta/2/alpha/gamma(1/beta)) + log2(exp(1))/numpoints * bb;
