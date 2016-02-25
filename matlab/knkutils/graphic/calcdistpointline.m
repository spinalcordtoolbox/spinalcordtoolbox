function f = calcdistpointline(pp,x,y)

% function f = calcdistpointline(pp,x,y)
%
% <pp> is a 2-element vector that specifies a line (using polyval's convention)
% <x>,<y> are matrices with x- and y-values
%
% return the perpendicular distance from the points
% given by <x> and <y> to the line given by <pp>.
%
% example:
% xx = [1 2 3];
% yy = [1 1 1];
% dists = calcdistpointline([1 0],xx,yy);
% figure; hold on;
% plot([-3 3],[-3 3],'k-');
% scatter(xx,yy,'ro');
% title(sprintf('%.5f ',dists));
% axis equal;

f = abs(polyval(pp,x)-y) / sqrt(pp(1)^2+1);
