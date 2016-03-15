function h = drawarrow(pt1,pt2,linestyle,ang,len,varargin)

% function h = drawarrow(pt1,pt2,linestyle,ang,len,varargin)
%
% <pt1> is [X0 Y0].  can have multiple rows for multiple cases.
% <pt2> is [X1 Y1].  can have multiple rows (number should match that of <pt1>).
% <linestyle> (optional) is like 'r-'.  default: 'r-'.
% <ang> (optional) is the number of degrees in the angle that the
%   sides of the arrow make with the arrow itself.  default: 30.
% <len> (optional) is the length of the sides of the arrow in units
%   of points.  default: 20.
% <varargin> (optional) are additional arguments to plot.m
%
% draw an arrow that starts at <pt1> and ends at <pt2>.
% return the handle to the line object(s) that we create.
% note that the arrow that we draw is tied to the axis range
% and axis position of the figure.  if you resize the figure,
% the arrowheads may become warped.
%
% example:
% figure; drawarrow([1 0],[3 10]);

% input
if ~exist('linestyle','var') || isempty(linestyle)
  linestyle = 'r-';
end
if ~exist('ang','var') || isempty(ang)
  ang = 30;
end
if ~exist('len','var') || isempty(len)
  len = 20;
end

% prep figure
hold on;

% plot temporary line to figure out what axes we are using
h = [];
for p=1:size(pt1,1)
  h(p) = plot([pt1(p,1) pt2(p,1)],[pt1(p,2) pt2(p,2)],linestyle,varargin{:});
end
ax = axis;
  prev = get(gca,'Units');  % store old
set(gca,'Units','points');
axpos = get(gca,'Position');
  set(gca,'Units',prev);
delete(h);

% define functions that convert points in the data space [X Y] to points in the screen space [X0 Y0]
datatoscreen = @(x) [axpos(1) + axpos(3) * (x(:,1)-ax(1)) / (ax(2)-ax(1)) ...
                     axpos(2) + axpos(4) * (x(:,2)-ax(3)) / (ax(4)-ax(3))];
screentodata = @(x) [ax(1) + (ax(2)-ax(1)) * ((x(:,1)-axpos(1)) / axpos(3)) ...
                     ax(3) + (ax(4)-ax(3)) * ((x(:,2)-axpos(2)) / axpos(4))];

% compute screen coordinates
pt1s = feval(datatoscreen,pt1);
pt2s = feval(datatoscreen,pt2);

% ok, continue
h = [];
for p=1:size(pt1,1)
  
  % figure out angle of vector
  temp = pt2s(p,:)-pt1s(p,:);
  angvector = atan2(temp(2),temp(1));
  
  % compute arrow sides as vectors and add it to the endpoint (in screen coordinates)
  arrowsides = len*ang2complex([angvector+pi-ang/180*pi angvector+pi+ang/180*pi]);
  arrowsidesfinal = repmat(pt2s(p,1) + j*pt2s(p,2),[1 2]) + arrowsides;
  
  % the final x and y coordinates in screen coordinates
  xx = [pt1s(p,1) pt2s(p,1) real(arrowsidesfinal(1)) pt2s(p,1) real(arrowsidesfinal(2))];
  yy = [pt1s(p,2) pt2s(p,2) imag(arrowsidesfinal(1)) pt2s(p,2) imag(arrowsidesfinal(2))];
  
  % convert to data coordinates
  final = zeros(length(xx),2);
  for q=1:length(xx)
    final(q,:) = feval(screentodata,[xx(q) yy(q)]);
  end
  
  % do it
  h(p) = plot(final(:,1),final(:,2),linestyle,varargin{:});

end

% make sure the axis range is preserved from the beginning
axis(ax);
  



% OLD JUNK:
% % calculate length and angle of desired arrow
% len = sqrt(sum((pt2-pt1).^2));
% ang = atan2(pt2(2)-pt1(2),pt2(1)-pt1(1));
% 
% % define a basic arrow (stolen from feather.m)
% xx = [0 1 .8 1 .8]';
% yy = [0 0 .08 0 -.08].';
% arrow = xx + j*yy;  % 5 x 1, complex entries.  this arrow is unit-length.
% arrow = arrow * exp(j*ang);
% arrow = arrow - real(arrow) + real(arrow)*dar(1);
% arrow = arrow - j*imag(arrow) + j*imag(arrow)*dar(2);
% arrow = arrow * exp(-j*ang);
% arrow = arrow / real(arrow(2));
% 
% % scale arrow, rotate arrow, offset arrow
% f = (arrow * len) * exp(j*ang) + (pt1(1) + j*pt1(2));
% 
% % do it
% h = plot(real(f),imag(f),linestyle,varargin{:});
