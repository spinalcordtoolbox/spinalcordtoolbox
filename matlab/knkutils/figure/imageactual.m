function h = imageactual(im)

% function h = imageactual(im)
%
% <im> is the filename of an image
%
% in the current figure, display the image specified by <im> at its 
% native resolution (thus, avoiding downsampling or upsampling).
% this is accomplished by changing the position and size of the current
% figure and its axes.  return the handle of the created image.
%
% note that if <im> is an indexed image and does not have an 
% associated colormap, then we default to the colormap gray(256).
%
% history:
% - 2013/07/02 - change functionality and clean up
%
% example:
% imwrite(uint8(255*rand(100,100,3)),'temp.png');
% figure; imageactual('temp.png');

% read image
[im,cmap] = imread(im);
if isempty(cmap)
  cmap = gray(256);
end

% calc
r = size(im,1);
c = size(im,2);

% change figure position
set(gcf,'Units','points');
pos = get(gcf,'Position');
newx = pos(1) - (c/1.25 - pos(3))/2;
newy = pos(2) - (r/1.25 - pos(4))/2;
set(gcf,'Position',[newx newy c/1.25 r/1.25]);

% change axis position
set(gca,'Position',[0 0 1 1]);

% draw image and set axes
h = image(im);
if size(im,3) ~= 3
  colormap(cmap);
end
axis equal tight off;
