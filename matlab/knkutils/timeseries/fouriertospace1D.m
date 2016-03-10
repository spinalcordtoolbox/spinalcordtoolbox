function f = fouriertospace1D(flt,sz,wantcheck,mode)

% function f = fouriertospace1D(flt,sz,wantcheck,mode)
%
% <flt> is a 1 x N magnitude filter in the Fourier domain (corresponding to the output of fft)
% <sz> is the desired filter size in number of time points (must be odd and less than
%   or equal to length(<flt>)).  special case is 0 which means to omit the 
%   truncation and hanning-windowing.  if negative, this means do not window and just truncate.
% <wantcheck> (optional) is whether to show some diagnostic figures. default: 0.
% <mode> (optional) is
%  -1 means do nothing.
%   0 means normalize filter to be unit-length
%   1 means normalize filter to sum to 1
%   default: 0.
%
% return a space-domain zero-phase filter.
% after construction of the filter, the filter is truncated 
% according to <sz>, hanning-windowed, and then normalized according to <mode>.
%
% note that truncation and hanning-windowing
% cause the frequency response of the filter is be imperfect.
%
% example:
% flt = fouriertospace1D(constructbutterfilter1D(100,20,5),21,1);

% SEE ALSO FOURIERTOSPACE.M

% input
if ~exist('wantcheck','var') || isempty(wantcheck)
  wantcheck = 0;
end
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end

% calc
res = length(flt);
if sz ~= 0
  n = (abs(sz)-1)/2;  % number of pixels extending in each direction beyond center
  assert(isint(n));
end

% construct filter and then fftshift so it is nice and centered
flt2 = fftshift(real(ifft(flt)));

% extract a small part of it and hanning window it
if sz == 0
  f = flt2;
else
  c = choose(mod(res,2)==0,res/2+1,(res+1)/2);
  if sz > 0
    f = flt2(c-n:c+n) .* hanning(sz)';
  else
    f = flt2(c-n:c+n);
  end
end

% normalize
switch mode
case -1
case 0
  f = unitlength(f);
case 1
  f = f/sum(f(:));
end

% show figures
if wantcheck
  % TODO: MAKE NICER (e.g. viewimage)
  figure; plot(flt2); title('original filter');
  figure; plot(fftshift(abs(fft(flt2)))); title('spectrum of original filter');
  figure; plot(f); title('new filter');
  figure; plot(fftshift(abs(fft(f)))); title('spectrum of new filter');
end
