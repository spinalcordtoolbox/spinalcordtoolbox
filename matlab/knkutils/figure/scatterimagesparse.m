function h = scatterimagesparse(x,y,im,rng,num,wanttitle,sc,ax)

% function h = scatterimagesparse(x,y,im,rng,num,wanttitle,sc,ax)
%
% <x>,<y> are vectors of the same size
% <im> is N x N x P where P is the length of <x>
% <rng> is [A B] with the range to use in imagesc.
% <num> (optional) is the number of points to show.  default: 1000.
% <wanttitle> (optional) is whether to change the title.  default: 1.
% <sc> (optional) is a scale factor for the size of the images.  default: 1.
% <ax> (optional) is an axis range to impose
%
% in existing figure window (if any), clear the figure and then
% scatter plot <x> against <y>, with a maximum of <num> points shown.
% this routine is deterministic in which points are (randomly) picked.
% return the handles.
%
% example:
% figure; scatterimagesparse(randn(1,100),randn(1,100),randn(100,100,100),[-2 2],10,1,2);

% constants
frac = .025;  % how much of xrange should the x-size of an image be?

% input
if ~exist('num','var') || isempty(num)
  num = 1000;
end
if ~exist('wanttitle','var') || isempty(wanttitle)
  wanttitle = 1;
end
if ~exist('sc','var') || isempty(sc)
  sc = 1;
end
if ~exist('ax','var') || isempty(ax)
  ax = [];
end

% figure out which ones
[xtemp,idx] = picksubset(x,num);
ytemp = y(idx);
imtemp = im(:,:,idx);

% pre do it (figure out dar and pos)
clf;
scatter(xtemp,ytemp);
if ~isempty(ax)
  axis(ax);
end
ax = axis;
dar = get(gca,'DataAspectRatio');
pos = getfigurepos(gcf,'points');
clf;

% prep axis
axis(ax);
set(gca,'YDir','normal');

% figure it out
xep = (ax(2)-ax(1)) * frac * sc;
yep = xep * dar(2)/dar(1) * pos(3)/pos(4);

% do it
hold on;
h = [];
for p=1:length(idx)
  h(p) = imagesc([xtemp(p)-xep/2 xtemp(p)+xep/2],[ytemp(p)+yep/2 ytemp(p)-yep/2],imtemp(:,:,p),rng);  % note the sign flip
end
axis(ax);
if wanttitle
  title(sprintf('showing %.2f percent',length(xtemp)/length(x(:))*100));
end
