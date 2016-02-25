function h = scattersparse(x,y,num,wanttitle,markersize,markercolor,marker)

% function h = scattersparse(x,y,num,wanttitle,markersize,markercolor,marker)
%
% <x>,<y> are matrices of same size
% <num> (optional) is the number of points to show.  default: 1000.
% <wanttitle> (optional) is whether to change the title.  default: 1.
% <markersize> (optional).  default: 16.
% <markercolor> (optional).  default: 'r'.
%   can also be a matrix the same size as <x> and <y>.
% <marker> (optional).  default: '.'.
%
% in existing figure window (if any), scatter plot <x> against <y>,
% with a maximum of <num> points shown.  this routine is deterministic
% in which points are (randomly) picked.  return the handles.
%
% example:
% figure; scattersparse(randn(1,1000),randn(1,1000),100,1);

% deal with input
if ~exist('num','var') || isempty(num)
  num = 1000;
end
if ~exist('wanttitle','var') || isempty(wanttitle)
  wanttitle = 1;
end
if ~exist('markersize','var') || isempty(markersize)
  markersize = 16;
end
if ~exist('markercolor','var') || isempty(markercolor)
  markercolor = 'r';
end
if ~exist('marker','var') || isempty(marker)
  marker = '.';
end

% figure out which ones
[xtemp,idx] = picksubset(x,num);
ytemp = y(idx);
if ~ischar(markercolor)
  markercolor = markercolor(idx);
end

% do it
h = scatter(xtemp,ytemp,markersize,markercolor,marker);
if wanttitle
  title(sprintf('showing %.2f percent',length(xtemp)/length(x(:))*100));
end
