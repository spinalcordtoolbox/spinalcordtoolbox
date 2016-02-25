function f = constructcosinefilter(res,cutoff,band)

% function f = constructcosinefilter(res,cutoff,band)
%
% <res> is the number of pixels along one side
% <cutoff> is
%   A means low-pass filter cutoff (in cycles per field-of-view)
%  -B means high-pass filter cutoff (in cycles per field-of-view)
%  [A B] means band-pass filter cutoffs (in cycles per field-of-view)
% <band> is width of the band over which the filter moves from 0 to 1
%
% use half of a cosine function to progress from 0 to 1.  also referred to
% as "tukey" or "tapered cosine"?  return the 2D magnitude filter in 
% Fourier space (not fftshifted).  the range of values is [0,1].  
% the result is suitable for use with imagefilter.m.
%
% example:
% a = randn(512,512);
% figure; imagesc(a); axis equal tight;
% figure; imagesc(fftshift(abs(fft2(a)))); axis equal tight;
% b = imagefilter(a,constructcosinefilter(512,[20 40],5));
% figure; imagesc(b); axis equal tight;
% figure; imagesc(fftshift(abs(fft2(b)))); axis equal tight;

% band-pass case
if length(cutoff) > 1

  % low-pass filter with cutoff at high value minus low-pass filter with cutoff at low value
  f = constructcosinefilter(res,cutoff(2),band) - constructcosinefilter(res,cutoff(1),band);

% low-pass case
elseif cutoff > 0

  % prep
  [xx,yy] = calccpfov(res);
  radius = sqrt(xx.^2+yy.^2);  % cycles per field-of-view
  f = zeros(size(radius));
  
  % figure out regions (see makecircleimage.m for similar idea)
  region1 = radius < cutoff-band/2;
  region2 = radius >= cutoff-band/2 & radius <= cutoff+band/2;
  region3 = radius > cutoff+band/2;
  
  % do it
  f(region1) = 1;
  f(region2) = cos((radius(region2)-(cutoff-band/2)) * pi/band)/2 + 0.5;
  f(region3) = 0;
  f = ifftshift(f);

% high-pass case
else

  % take one minus a low-pass filter with cutoff at the specified value
  f = 1 - constructcosinefilter(res,-cutoff,band);

end
