function f = calccpfov1D(n,mode)

% function f = calccpfov1D(n,mode)
%
% <n> is the number of time points
% <mode> (optional) is
%   0 means return in units of cycles per FOV
%   1 means return in units of [-pi,pi)
%   default: 0.
%  
% return the number of cycles per field-of-view corresponding 
% to the output of fft after fftshifting.
%
% example:
% figure; bar(calccpfov1D(32));

% SEE ALSO CALCCPFOV.M

% input
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end

% construct range
if mod(n,2)==0
  f = -n/2:n/2-1;
else
  f = -(n-1)/2:(n-1)/2;
end

% convert if necessary
if mode==1
  f = f / (n/2) * pi;
end
