function f = fftshift2(x)

% function f = fftshift2(x)
%
% <x> is a matrix
%
% apply fftshift along the first and then second dimensions.
%
% example:
% im = cat(3,getsampleimage,getsampleimage(2));
% figure; imagesc(makeimagestack(im));
% figure; imagesc(makeimagestack(fftshift2(im)));

f = fftshift(fftshift(x,1),2);
