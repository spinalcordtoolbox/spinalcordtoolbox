function drawsector(res,theta1,theta2,r1,r2,color,bg,granularity)

% function drawsector(res,theta1,theta2,r1,r2,color,bg,granularity)
%
% <res> is
%   0 means standard figure coordinate frame
%  -1 means y-reversed coordinate frame
% <theta1> is an angle in [0,2*pi)
% <theta2> is an angle in [0,2*pi]
% <r1> is a radius in [0,1].
%   1 means all the way from the center to the edge of the image.
% <r2> is a radius in [0,1].
%   1 means all the way from the center to the edge of the image.
% <color> is a 3-element vector with the sector color
% <bg> is a 3-element vector with the background color.
%   [] means do not draw the background.
% <granularity> (optional) is how many points in a complete revolution.  default: 360.
%
% draw a sector on the current figure.  we automatically set 
% the axis bounds and also reverse the y-axis if necessary.
%
% example:
% figure; drawsector(0,.1,pi/4,.5,1,[1 0 0],[0 0 0]); axis equal tight;

% input
if ~exist('granularity','var') || isempty(granularity)
  granularity = 360;
end

% prep figure
hold on;

% draw square for background
if ~isempty(bg)
  hbg = patch([-.5 -.5 .5 .5 -.5],[-.5 .5 .5 -.5 -.5],bg);
  set(hbg,'EdgeColor','none');
end

% calc
numedges = round(mod2(theta2-theta1,2*pi)/(2*pi)*granularity);

% deal with r interpretation
r1 = r1 * 0.5;
r2 = r2 * 0.5;

% go out
thetas = [theta1 theta1];
rs = [r1 r2];

% going around the outside
thetas = [thetas theta1+(1:(numedges-1))/numedges*(theta2-theta1)];
rs = [rs repmat(r2,1,numedges-1)];

% come in
thetas = [thetas theta2 theta2];
rs = [rs r2 r1];

% go around the inside
thetas = [thetas theta2+(1:(numedges-1))/numedges*(theta1-theta2)];
rs = [rs repmat(r1,1,numedges-1)];

% draw
[X,Y] = pol2cart(thetas,rs);
h = patch(X,Y,color);
set(h,'EdgeColor','none');

% prep figure
axis([-.5 .5 -.5 .5]);
if res ~= 0
  set(gca,'YDir','reverse');
end
