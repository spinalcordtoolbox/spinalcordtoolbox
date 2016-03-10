function f = constructbutterfilter1D(n,cutoff,order)

% function f = constructbutterfilter1D(n,cutoff,order)
%
% <n> is the number of time points
% <cutoff> is
%   A means low-pass filter cutoff (in cycles per field-of-view).  can be Inf.
%  -B means high-pass filter cutoff (in cycles per field-of-view)
%  [A B] means band-pass filter cutoffs (in cycles per field-of-view).  special case
%    is when A is 0, which means to only low-pass filter.
% <order> (optional) is a positive integer indicating the order of the Butterworth filter.
%   default: 5.
%
% return the 1D magnitude filter in Fourier space (not fftshifted).
% the range of values is [0,1].  the result is suitable for use with tsfilter.m.
%
% example:
% a = randn(1,128);
% b = tsfilter(a,constructbutterfilter1D(128,[20 40],5));
% figure; hold on; plot(a,'r-'); plot(b,'b-');
% figure; hold on; plot(fftshift(abs(fft(a))),'r-'); plot(fftshift(abs(fft(b))),'b-');

% SEE ALSO CONSTRUCTBUTTERFILTER.M

% input
if ~exist('order','var') || isempty(order)
  order = 5;
end

% band-pass case
if length(cutoff) > 1

  % low-pass filter with cutoff at high value minus low-pass filter with cutoff at low value
  f = constructbutterfilter1D(n,cutoff(2),order);
  if cutoff(1) ~= 0
    f = f - constructbutterfilter1D(n,cutoff(1),order);
  end

% low-pass case
elseif cutoff > 0

  % do it
  radius = calccpfov1D(n);  % cycles per field-of-view
  f = ifftshift(sqrt(1./(1+(radius./cutoff).^(2*order))));

% high-pass case
else

  % take one minus a low-pass filter with cutoff at the specified value
  f = 1 - constructbutterfilter1D(n,-cutoff,order);

end
