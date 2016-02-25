function f = generaterandomphase(res,num)

% function f = generaterandomphase(res,num)
%
% <res> is the number of pixels on a side (must be at least 3)
% <num> (optional) is the number of images desired.  default: 1.
%  
% return a <res> x <res> x <num> matrix with elements that are unit-length complex
% numbers.  this matrix is ready for multiplication with the output of fft2.
% the result is to randomly perturb the phase of each Fourier component of each image.
% note that some of the Fourier components (e.g. the DC component) are special in that
% the complex numbers corresponding to these components are restricted to be either 1 or -1,
% since these components have no imaginary part.
%
% example:
% a = randn(5,5);
% b = fft2(a) .* generaterandomphase(5);
% c = ifft2(b);
% allzero(imag(c))

% input
if ~exist('num','var') || isempty(num)
  num = 1;
end

% get out!
if res < 3
  error('<res> must be at least 3');
end

% do it
if mod(res,2)==0

  % start by doing the two weird legs
  f = zeros(2,res-1,num);                             % initialize in convenient form
  f(:,1:(res-2)/2,:) = rand(2,(res-2)/2,num)*(2*pi);  % fill in the first half with random phase in [0,2*pi]
  f = f + -flipdim(f,2);                              % symmetrize (the mirror gets the negative phase)
  f(:,(res-2)/2+1,:) = (rand(2,1,num)>.5)*pi;         % fill in the center (either 0 or pi)
  
  % ok, put it all together
  f = [(rand(1,1,num)>.5)*pi            f(1,:,:);
       reshape(f(2,:,:),[res-1 1 num])  helper(res-1,num)];

else
  f = helper(res,num);
end

% convert to imaginary numbers
f = exp(j*f);

% unshift
f = ifftshift2(f);  % use ifftshift with a dimension specified because f might have multiple elements in the third dimension

%%%%%

function f = helper(res,num)

% return a matrix of dimensions <res> x <res> x <num> with appropriate
% random phase values in [0,2*pi].  note that the center (DC component)
% has phase values that are either 0 or pi.  the returned matrix is
% as if fftshift has been called.

f = zeros(res*res,num);           % initialize in convenient form
nn = (res*res-1)/2;               % how many in first half?
f(1:nn,:) = rand(nn,num)*(2*pi);  % fill in the first half with random phase in [0,2*pi]
f = reshape(f,[res res num]);     % reshape
f = f + -flipdim(flipdim(f,1),2); % symmetrize (the mirror gets the negative phase)
f((res+1)/2,(res+1)/2,:) = (rand(1,1,num)>.5)*pi;  % fill in the center (either 0 or pi)
