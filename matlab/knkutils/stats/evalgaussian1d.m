function f = evalgaussian1d(params,x)

% function f = evalgaussian1d(params,x)
%
% <params> is [m s g d] where
%   <m> is the mean
%   <s> is the standard deviation
%   <g> is the gain
%   <d> is the offset
% <x> is a matrix containing x-coordinates to evaluate at.
% 
% evaluate the 1D Gaussian at <x>.
%
% the FWHM is 2*sqrt(2*log(2))*<s>.  if you want a FWHM of X, 
% you should set <s> to X/(2*sqrt(2*log(2))).
%
% assuming <d> is 0, if you want the area under the curve to be 1,
% you should set <g> to 1/(s*sqrt(2*pi)).
% 
% example:
% xx = 0:.01:1;
% yy = evalgaussian1d([.5 .1 2 0],xx);
% figure; plot(xx,yy,'ro-');

% input
m = params(1);
s = params(2);
g = params(3);
d = params(4);

% do it
f = g*exp( (x-m).^2/-(2*s^2) ) + d;
