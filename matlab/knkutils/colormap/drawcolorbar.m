function h = drawcolorbar(range,ticks,cmap,str,orient)

% function h = drawcolorbar(range,ticks,cmap,str,orient)
%
% <range> is the range like [0 1]
% <ticks> are the desired tick values like 0:.1:1.  can be [].
% <cmap> is the colormap
% <str> is the string label to put next to the colorbar.  can be [].
% <orient> is 0 means vertical, 1 means horizontal
%
% draw an image and a colorbar on the current figure.
% tick marks are omitted from the colorbar.
% return the handle to the colorbar.
%
% example:
% figure; drawcolorbar([0 10],0:3:9,hot(10),'Test',1);

% calc
v = choose(orient==0,'Y','X');
w = choose(orient==0,'vert','horiz');

% do it
hold on;
caxis(range);
imagesc([range; fliplr(range)],range); axis equal tight;
axis ij;
colormap(cmap);
h = colorbar(w);
set(h,[v,'Tick'],ticks,[v,'Lim'],range,'TickLength',[0 0]);
set(get(h,[v,'Label']),'string',str);
axis off;
