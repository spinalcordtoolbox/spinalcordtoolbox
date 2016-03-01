function f = evaldivnorm(params,x)

% function f = evaldivnorm(params,x)
%
% <params> is [n s g d] where
%   <n> is the exponent
%   <s> is the sigma.  special case is Inf which means to omit the denominator.
%   <g> is the gain
%   <d> is the offset
% <x> is a matrix containing x-coordinates to evaluate at.
% 
% evaluate the divisive-normalization function at <x>.
% the function is g * (x.^n ./ (x.^n + s.^n)) + d.
%
% note that there is some overlap with divnorm.m.
%
% example:
% xx = 0:.01:1;
% yy = evaldivnorm([2 .4 1 0],xx);
% figure; plot(xx,yy,'ro-');

% input
n = params(1);
s = params(2);
g = params(3);
d = params(4);

% do it
f = divnorm(x,n,s);
if g ~= 1
  f = g*f;
end
if d ~= 0
  f = f + d;
end
