function f = constructorientationfilter(res,or,sigma)

% function f = constructorientationfilter(res,or,sigma)
%
% <res> is the number of pixels along one side
% <or> is an orientation in [0,pi)
% <sigma> is the desired orientation standard deviation
%
% use a von Mises distribution to construct an orientation filter.
% return a 2D magnitude filter in Fourier space (not fftshifted).
% the range of values is [0,1].  the result is suitable for use with 
% imagefilter.m.
%
% a value of 0 is returned for the DC component.
%
% history:
% 2011/04/25 - fix major bug. orientation was interpreted incorrectly.
%
% example:
% a = randn(64,64);
% viewimage(a);
% b = imagefilter(a,constructorientationfilter(64,pi/4,pi/8));
% viewimage(b);

%% WORRY ABOUT IMAGINARY PART WHEN IFFT2????

[xx,yy] = calccpfov(res);
ors = mod(atan2(yy,xx),pi);
k = 1/sigma^2;
f = exp(k*cos(2*mod(pi/2-(ors+or),pi))) / exp(k);
f(xx==0 & yy==0) = 0;
f = ifftshift(f);
