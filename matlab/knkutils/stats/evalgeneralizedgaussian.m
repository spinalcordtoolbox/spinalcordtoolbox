function f = evalgeneralizedgaussian(params,x)

% function f = evalgeneralizedgaussian(params,x)
%
% <params> is [mn alpha beta] where
%   mn is the mean
%   alpha is the scaling (positive)
%   beta is the exponent (positive)
% <x> is a matrix of values to evaluate at.
%
% evaluate the generalized Gaussian at <x>.
%
% example:
% xx = -10:.01:10;
% yy = evalgeneralizedgaussian([0 1 1],xx);
% figure; scatter(xx,yy,'r.');
% sum(yy*.01)

% input
mn = params(1);
alpha = params(2);
beta = params(3);

% do it
f = (beta/2/alpha/gamma(1/beta)) * exp(abs(x-mn).^beta/(-alpha^beta));
