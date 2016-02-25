function imagesc3D(im,cmap)

% function fig = viewimages(im)
% 
% <im> is a set of 2D images (res x res x images)
%
% make a figure window and show all of the images.
% return the figure number.

setfigurepos([50 50 500 500]);
if nargin<2
    imagesc(makeimagestack(squeeze(im))); axis equal tight;
else
    imagesc(makeimagestack(squeeze(im)),cmap); axis equal tight;
end

drawnow;
