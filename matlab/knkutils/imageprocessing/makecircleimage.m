function [f,xx,yy] = makecircleimage(res,r,xx,yy,r2,mode,center,sc)

% function [f,xx,yy] = makecircleimage(res,r,xx,yy,r2,mode,center,sc)
%
% <res> is the number of pixels along one side
% <r> is size of radius in pixels
% <xx>,<yy> (optional) are speed-ups (dependent on <res>)
% <r2> (optional) is size of next radius in pixels.  default: <r>.
% <mode> (optional) is
%   0 means normal
%   1 means use absolute value of x-direction for the "radius"
%   2 means use absolute value of y-direction for the "radius"
%   3 means use x-direction for the "radius".  in this case,
%     you can interpret <r> and <r2> as signed coordinate values.
%   4 means use y-direction for the "radius".  in this case,
%     you can interpret <r> and <r2> as signed coordinate values.
%   default: 0.
% <center> (optional) is [R C] with the row and column indices of
%   the center.  can be decimal.  default: [(1+<res>)/2 (1+<res>)/2].
% <sc> (optional) is [SCX SCY] with the scale factor along the x- and
%   y-directions.  for example, [0.5 0.9] means to shrink the x-direction
%   by 50% and shrink the y-direction by 10%.  this option takes effect
%   only when <mode> is 0.  default: [1 1].
%
% the image is a white circle (1) on a black background (0).
%   (when <mode> is 1-4, the image is a white rectangle (1)
%   on a black background (0).)
% if <r2> is not supplied, we return a binary image.
% if <r2> is supplied, we gradually ramp from white to black
%   using a cosine function.  note that <r2> being not supplied
%   is equivalent to supplying <r> for <r2>.
%
% example:
% figure; imagesc(makecircleimage(100,20,[],[],40)); axis equal tight;

% construct coordinates
if ~exist('xx','var') || isempty(xx)
  [xx,yy] = calcunitcoordinates(res);
end
if ~exist('r2','var') || isempty(r2)
  r2 = r;
end
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end
if ~exist('center','var') || isempty(center)
  center = [(1+res)/2 (1+res)/2];
end
if ~exist('sc','var') || isempty(sc)
  sc = [1 1];
end

% calc
r = r/res;
r2 = r2/res;
center = (center-(1+res)/2) * (1/res);
center(1) = -center(1);

% figure out regions
switch mode
case 0
  radius = sqrt((xx-center(2)).^2/sc(1).^2 + (yy-center(1)).^2/sc(2).^2);
case 1
  radius = abs((xx-center(2)));
case 2
  radius = abs((yy-center(1)));
case 3
  radius = xx-center(2);
case 4
  radius = yy-center(1);
end
region1 = radius <= r;
region2 = radius > r & radius <= r2;
region3 = radius > r2;

% do it
f = zeros(res,res);
f(region1) = 1;
f(region2) = cos((radius(region2)-r) * pi/(r2-r))/2 + 0.5;
f(region3) = 0;
