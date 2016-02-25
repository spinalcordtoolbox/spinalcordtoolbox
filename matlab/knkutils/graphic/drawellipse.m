function h = drawellipse(x,y,ang,sd1,sd2,theta0,theta1,linestyle,granularity)

% function h = drawellipse(x,y,ang,sd1,sd2,theta0,theta1,linestyle,granularity)
%
% <x> is x-position of ellipse center
% <y> is y-position of ellipse center
% <ang> is the orientation in [0,2*pi).  0 means major axis is parallel to x-axis.
% <sd1> is the std dev along the major axis
% <sd2> is the std dev along the minor axis
% <theta0> (optional) is the starting angle in [0,2*pi).  default: 0.
% <theta1> (optional) is the ending angle in [0,2*pi].  default: 2*pi.
% <linestyle> (optional) is like 'r-'.  default: 'r-'.
%   special case is {C} where C is a color char, scalar, or vector.
%   in this case, a patch object is created instead of a line object.
% <granularity> (optional) is how many points in a complete revolution.  default: 360.
%
% draw a complete or partial ellipse on the current figure.  the ellipse corresponds 
% to +/- 1 standard deviation along the major and minor axes.  we proceed CCW from 
% <theta0> to <theta1>.  return the handle to the line object that we create.
%
% example:
% figure; drawellipse(3,1,pi/6,3,1,0,3*pi/2,'ro-',50); axis equal;

% input
if ~exist('theta0','var') || isempty(theta0)
  theta0 = 0;
end
if ~exist('theta1','var') || isempty(theta1)
  theta1 = 2*pi;
end
if ~exist('linestyle','var') || isempty(linestyle)
  linestyle = 'r-';
end
if ~exist('granularity','var') || isempty(granularity)
  granularity = 360;
end

% deal with thetas for wrap-around case
if theta1 < theta0
  theta1 = theta1 + 2*pi;
end

% prep figure
hold on;

% figure out thetas that we want
thetas = linspace(theta0,theta1,ceil((theta1-theta0)/(2*pi) * (granularity+1)));

% figure out the base coordinates
[X,Y] = pol2cart(thetas,1);
coord = [X; Y];

% scale
coord = diag([sd1 sd2]) * coord;

% rotate   [cos(ang) sin(ang); -sin(ang) cos(ang)] rotates CW
coord = [cos(-ang) sin(-ang); -sin(-ang) cos(-ang)]*coord;

% translate     [TODO: TRANSFORMATIONS SHOULD BE CLEANED UP AND MADE INTO FUNCTIONS!]
coord(1,:) = coord(1,:) + x;
coord(2,:) = coord(2,:) + y;

% do it
if iscell(linestyle)
  h = patch(coord(1,:),coord(2,:),linestyle{1});
else
  h = plot(coord(1,:),coord(2,:),linestyle);
end
