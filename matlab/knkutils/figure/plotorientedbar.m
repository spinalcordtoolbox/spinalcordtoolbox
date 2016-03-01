function h = plotorientedbar(x,y,or,len,width,style)

% function h = plotorientedbar(x,y,or,len,width,style)
%
% <x>,<y> are the x- and y-coordinates
% <or> is the orientation in [0,pi)
% <len> is the length of the bar
% <width> is the 'LineWidth' of the bar
% <style> is like 'r-'
%
% return the handle to a line object.
%
% example:
% figure; plotorientedbar(1,0,pi/6,2,2,'r-'); axis equal; axis([-3 3 -3 3]);

[dx,dy] = pol2cart(or,len/2);
h = plot([x-dx x+dx],[y-dy y+dy],style,'LineWidth',width);
