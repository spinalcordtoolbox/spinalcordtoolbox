function setfigurepos(fig,pos)

% function setfigurepos(fig,pos)
%
% <fig> (optional) is the figure handle.  can be a vector.  default: [gcf].
% <pos> is one of the following:
%   (1) the position in normalized units
%   (2) a scalar which indicates the scale factor to apply (anchoring the figure at the center)
%   (3) the position in points units
%   we decide that a case is (3) if any value is greater than 10.
%
% set the position of <fig> without mangling the 'Units' setting.
%
% example:
% figure; setfigurepos(1.5);

% input
if nargin==1
  pos = fig;
  fig = [gcf];
end

% do it
for p=1:length(fig)
  
  % store old
  prev = get(fig(p),'Units');

  % the scale factor case
  if length(pos)==1
    set(fig(p),'Units','normalized');
    oldpos = getfigurepos(fig(p));
    newsize = pos*oldpos(3:4);
    set(fig(p),'Position',[oldpos(1:2) - (newsize-oldpos(3:4))/2 newsize]);

  % the points case
  elseif any(pos>10)
    set(fig(p),'Units','points');
    set(fig(p),'Position',pos);

  % the normalized case
  else
    set(fig(p),'Units','normalized');
    set(fig(p),'Position',pos);
  end

  % recall old
  set(fig(p),'Units',prev);

end
