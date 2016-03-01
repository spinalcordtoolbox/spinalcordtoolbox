function f = fouriertospace(flt,sz,wantcheck,mode)

% function f = fouriertospace(flt,sz,wantcheck,mode)
%
% <flt> is an N x N magnitude filter in the Fourier domain (corresponding to the output of fft2)
% <sz> is the desired filter size in number of pixels (must be odd and less than
%   or equal to size(<flt>,1)).  special case is 0 which means to omit the 
%   truncation and hanning-windowing.  if negative, this means do not window and just truncate.
% <wantcheck> (optional) is whether to show some diagnostic figures. default: 0.
% <mode> (optional) is
%   0 means normalize filter to be unit-length
%   1 means normalize filter to sum to 1
%   default: 0.
%
% return a space-domain zero-phase image filter.
% after construction of the filter, the filter is truncated 
% according to <sz>, hanning-windowed, and then normalized according to <mode>.
%
% note that truncation and hanning-windowing
% cause the frequency response of the filter is be imperfect.
%
% example:
% flt = fouriertospace(constructbutterfilter(100,10,5),21,1);

% SEE ALSO FOURIERTOSPACE1D.M

% input
if ~exist('wantcheck','var') || isempty(wantcheck)
  wantcheck = 0;
end
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end

% calc
res = size(flt,1);
if sz ~= 0
  n = (abs(sz)-1)/2;  % number of pixels extending in each direction beyond center
  assert(isint(n));
end

% construct filter and then fftshift so it is nice and centered
flt2 = fftshift(real(ifft2(flt)));

% extract a small part of it and hanning window it
if sz == 0
  f = flt2;
else
  c = choose(mod(res,2)==0,res/2+1,(res+1)/2);
  if sz > 0
    f = flt2(c-n:c+n,c-n:c+n) .* (hanning(sz)*hanning(sz)');
  else
    f = flt2(c-n:c+n,c-n:c+n);
  end
end

% normalize
if mode==0
  f = unitlength(f);
else
  f = f/sum(f(:));
end

% show figures
if wantcheck
  figure; viewimage(flt2); fprintf('The first figure has the original filter.\n');
  figure; viewimage(placematrix(zeros(res,res),f,[])); fprintf('The second figure has the new filter padded with zeros.\n');
  figure; viewimage(f); fprintf('The third figure has the new filter not padded.\n');
end
