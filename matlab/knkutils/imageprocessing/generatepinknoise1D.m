function f = generatepinknoise1D(n,alpha,num,mode)

% function f = generatepinknoise1D(n,alpha,num,mode)
%
% <n> is the number of time points (must be at least 3)
% <alpha> (optional) is the exponent to apply to the
%   amplitude spectrum (i.e. 1/f^alpha).  default: 1.
% <num> (optional) is the number of time-series desired.  default: 1.
% <mode> (optional) is
%   0 means fixed amplitude spectrum + random phase
%   1 means white Gaussian noise multiplied by desired amplitude spectrum
%   default: 0.
%
% generate samples of pink-noise time-series.
% each time-series is generated independently.
% the DC component is manually set to 0 (thus, each time-series is zero-mean).
% no post-hoc std dev normalization is performed.  when <mode> is 0,
% the std dev of each time-series is necessarily constant; when <mode> is 1, 
% the std dev of each time-series can vary.
%  
% return <f> with the various time-series (<n> x <num>).
%
% example:
% figure; plot(generatepinknoise1D(500,[],3));

% input
if ~exist('alpha','var') || isempty(alpha)
  alpha = 1;
end
if ~exist('num','var') || isempty(num)
  num = 1;
end
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end

% calculate 1/f amplitude matrix (for the DC component, manually set it to 0)
cpfov = ifftshift(calccpfov1D(n));
amp = zerodiv(1,abs(cpfov).^alpha);  % not fftshifted

switch mode
case 0
  f = real(ifft(repmat(amp',[1 num]) .* exp(j*angle(fft(randn(n,num),[],1))),[],1));
case 1
  f = real(ifft(fft(randn(n,num),[],1) .* repmat(amp',[1 num]),[],1));
end
