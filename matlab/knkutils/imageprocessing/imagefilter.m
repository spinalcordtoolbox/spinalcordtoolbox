function f = imagefilter(im,flt,mode)

% function f = imagefilter(im,flt,mode)
%
% <im> is an N x N square image.  can have multiple images along the third dimension.
% <flt> is a filter (in either the Fourier domain or space domain)
% <mode> (optional) is
%   0 means interpret <flt> as an N x N magnitude filter in the Fourier domain,
%     and do the filtering in the Fourier domain.
%   [1 sz] means interpret <flt> as an N x N magnitude filter in the Fourier domain,
%     but do the filtering in the space domain using imfilter.m and 'replicate'.
%     in order to convert the Fourier filter to the space domain, we use
%     fouriertospace.m and sz and make the filter unit-length (see fouriertospace.m 
%     for more details).
%   2 means interpret <flt> as a space-domain filter and do the filtering in the
%     space domain using imfilter.m and 'replicate'.
%   default: 0.
%
% return the filtered image(s).  we force the output to be real-valued.
% in general, beware of wraparound and edge issues!
%
% example:
% flt = zeros(100,100);
% flt(50:52,50:52) = 1;
% flt = ifftshift(flt);
% figure; imagesc(imagefilter(randn(100,100),flt));

% constants
num = 1000;  % number to do at a time

% input
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end

% construct space filter if necessary
if mode(1)==1
  flt = fouriertospace(flt,mode(2),[],0);
end

% do it
switch mode(1)
case 0
  f = [];
  for p=1:ceil(size(im,3)/num)
    mn = (p-1)*num+1;
    mx = min(size(im,3),(p-1)*num+num);
    f = cat(3,f,real(ifft2(fft2(im(:,:,mn:mx)).*repmat(flt,[1 1 mx-mn+1]))));
  end
case {1 2}
  f = processmulti(@imfilter,im,flt,'replicate','same','conv');
end
