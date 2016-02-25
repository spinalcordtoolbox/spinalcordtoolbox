function setaxispos(ax,pos)

% function setaxispos(ax,pos)
%
% <ax> (optional) is the axis handle.  can be a vector.  default: [gca].
% <pos> is one of the following:
%   (1) the position in normalized units
%   (2) a scalar which indicates the scale factor to apply
%
% set the position of <ax> without mangling the 'Units' setting.
%
% example:
% figure; scatter(randn(1,100),randn(1,100)); setaxispos(.5);

% input
if nargin==1
  pos = ax;
  ax = gca;
end

% do it
for p=1:length(ax)

  axsave = axis(ax(p));  % save the original limits
  prev = get(ax(p),'Units');  % store old
  set(ax(p),'Units','normalized');
  if isscalar(pos)
    oldpos = getfigurepos(ax(p));
    newsize = pos*oldpos(3:4);
    set(ax(p),'Position',[oldpos(1:2) - (newsize-oldpos(3:4))/2 newsize]);
  else
    set(ax(p),'Position',pos);
  end
  set(ax(p),'Units',prev);
  axis(ax(p),axsave);

end
