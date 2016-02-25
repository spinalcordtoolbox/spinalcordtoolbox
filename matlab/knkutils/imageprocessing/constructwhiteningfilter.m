function f = constructwhiteningfilter(res)

% function f = constructwhiteningfilter(res)
%
% <res> is the number of pixels along one side of the image
%
% return f/1 2D magnitude filter in Fourier space (not fftshifted).
% the result is suitable for use with imagefilter.m.
%
% example:
% a = getsampleimage;
% flt = constructwhiteningfilter(size(a,1));
% figure; imagesc(a); axis equal tight;
% figure; imagesc(imagefilter(a,flt)); axis equal tight;
% a = generatepinknoise(100);
% flt = constructwhiteningfilter(size(a,1));
% figure; imagesc(a); axis equal tight;
% figure; imagesc(imagefilter(a,flt)); axis equal tight;

% calc
[xx,yy] = calccpfov(res);
radius = sqrt(xx.^2+yy.^2);  % cycles per field-of-view

% construct filter
f = ifftshift(radius);
