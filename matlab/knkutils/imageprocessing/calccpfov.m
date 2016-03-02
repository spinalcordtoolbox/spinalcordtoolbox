function [xx,yy] = calccpfov(res)

% function [xx,yy] = calccpfov(res)
%
% <res> is the number of pixels on a side
%  
% return <xx> and <yy> which contain the number of cycles per field-of-view
% in the x- and y-directions, corresponding to the output of fft2 after 
% fftshifting.
%
% example:
% [xx,yy] = calccpfov(32);
% figure; imagesc(xx);
% figure; imagesc(yy);

% SEE ALSO CALCCPFOV1D.M

if mod(res,2)==0
  [xx,yy] = meshgrid(-res/2:res/2-1,-res/2:res/2-1);
else
  [xx,yy] = meshgrid(-(res-1)/2:(res-1)/2,-(res-1)/2:(res-1)/2);
end
