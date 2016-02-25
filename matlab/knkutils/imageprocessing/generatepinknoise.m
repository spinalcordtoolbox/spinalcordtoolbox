function f = generatepinknoise(res,alpha,num,mode)

% function f = generatepinknoise(res,alpha,num,mode)
%
% <res> is the number of pixels on a side (must be at least 3)
% <alpha> (optional) is the exponent to apply to the
%   amplitude spectrum (i.e. 1/f^alpha).  default: 1.
% <num> (optional) is the number of images desired.  default: 1.
% <mode> (optional) is
%   0 means fixed amplitude spectrum + random phase
%   1 means white Gaussian noise multiplied by desired amplitude spectrum
%   default: 0.
%
% generate samples of pink noise.  each image is generated independently.
% the DC component is manually set to 0 (thus, each image is zero-mean).
% no post-hoc std dev normalization is performed.  when <mode> is 0,
% the std dev of each image is necessarily constant; when <mode> is 1, 
% the std dev of each image can vary.
%  
% return <f> as images (<res> x <res> x <num>).
%
% example:
% figure; imagesc(generatepinknoise(100));

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
[xx,yy] = calccpfov(res);
cpfov = ifftshift(sqrt(xx.^2 + yy.^2));
amp = zerodiv(1,cpfov.^alpha);  % not fftshifted

switch mode
case 0
  f = real(ifft2(repmat(amp,[1 1 num]) .* generaterandomphase(res,num)));
case 1  % [see chandler 07]
  f = imagefilter(randn(res,res,num),amp);
end
