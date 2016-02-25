function im = detectedges(im,sd)

% function im = detectedges(im,sd)
%
% <im> is a square image.  can have multiple images along the third dimension.
% <sd> is the standard deviation of an isotropic 2D Gaussian filter
%
% detect edges by smoothing with a 2D Gaussian filter and then convolving with
% a horizontal derivative filter ([-1 0 1]) and a vertical derivative filter
% ([-1 0 1]').  the outputs of the two derivative filters are squared, summed,
% and square-rooted.
%
% note that this procedure is similar to the Sobel and Prewitt operators 
% for edge detection.
%
% example:
% figure; imagesc(detectedges(getsampleimage,2)); axis equal tight;

% smooth image with Gaussian
im = imagefilter(im,constructsmoothingfilter([sd sd],0.01),2);

% convolve with derivative filters and then sq, sum, sq-root
im = sqrt(imagefilter(im,unitlength([-1 0 1]),2).^2 + imagefilter(im,unitlength([-1 0 1])',2).^2);
