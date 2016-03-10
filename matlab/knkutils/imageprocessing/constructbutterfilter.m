function f = constructbutterfilter(res,cutoff,order)

% function f = constructbutterfilter(res,cutoff,order)
%
% <res> is the number of pixels along one side
% <cutoff> is
%   A means low-pass filter cutoff (in cycles per field-of-view)
%  -B means high-pass filter cutoff (in cycles per field-of-view)
%  [A B] means band-pass filter cutoffs (in cycles per field-of-view)
% <order> is a positive integer indicating the order of the Butterworth filter
%
% return the 2D magnitude filter in Fourier space (not fftshifted).
% the range of values is [0,1].  the result is suitable for use with imagefilter.m.
%
% example:
% a = randn(512,512);
% figure; imagesc(a); axis equal tight;
% figure; imagesc(fftshift(abs(fft2(a)))); axis equal tight;
% b = imagefilter(a,constructbutterfilter(512,[20 40],5));
% figure; imagesc(b); axis equal tight;
% figure; imagesc(fftshift(abs(fft2(b)))); axis equal tight;

% SEE ALSO CONSTRUCTBUTTERFILTER1D.M

% band-pass case
if length(cutoff) > 1

  % low-pass filter with cutoff at high value minus low-pass filter with cutoff at low value
  f = constructbutterfilter(res,cutoff(2),order) - constructbutterfilter(res,cutoff(1),order);

% low-pass case
elseif cutoff > 0

  % do it
  [xx,yy] = calccpfov(res);
  radius = sqrt(xx.^2+yy.^2);  % cycles per field-of-view
  f = ifftshift(sqrt(1./(1+(radius./cutoff).^(2*order))));

% high-pass case
else

  % take one minus a low-pass filter with cutoff at the specified value
  f = 1 - constructbutterfilter(res,-cutoff,order);

end
