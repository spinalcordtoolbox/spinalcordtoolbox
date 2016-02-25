function f = ifftshift2(x)

% function f = ifftshift2(x)
%
% <x> is a matrix
%
% apply ifftshift along the first and then second dimensions.
%
% example:
% im = cat(3,getsampleimage,getsampleimage(2));
% figure; imagesc(makeimagestack(im));
% figure; imagesc(makeimagestack(ifftshift2(im)));

f = ifftshift(ifftshift(x,1),2);
