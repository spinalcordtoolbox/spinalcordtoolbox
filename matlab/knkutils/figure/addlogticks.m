function addlogticks(wh)

% function addlogticks(wh)
%
% <wh> is 'x' or 'y' referring to the axis to modify
%
% the point of this function is to add log ticks for a 
% figure whose data have already been transformed by log.m.
% what we do is to set the XTick and XTickLabel (or YTick
% and YTickLabel) properties of the current axis appropriately.
%
% notes:
% - we do not achieve a shortening of the length of the minor ticks
%   (MATLAB does).
% - the numerical formatting of the tick labels may not be exactly 
%   the same as what MATLAB does.
% - we do not change the axis range of the figure (MATLAB does).
%
% example:
% x = rand(1,100); y = rand(1,100);
% figure; scatter(x,y); set(gca,'XScale','log');
% figure; scatter(log(x),y); addlogticks('x');

% figure out range
ax = axis;
if isequal(wh,'x')
  rng = ax(1:2);
else
  rng = ax(3:4);
end

% figure out liberal bounds
temp = log10(exp(rng));
tick = 10.^(floor(temp(1)):ceil(temp(2)));

% add entries for off terms
tick = upsamplematrix(tick,[1 9]);

% construct the labels
ticklabel = mat2cellstr(tick);

% get the ticks exactly right; get the ticklabels to be blank for off terms
for p=1:9:length(tick)
  tick(p+(0:8)) = (1:9) * tick(p);
  ticklabel(p+(1:8)) = repmat({''},[1 8]);
end

% set it
if isequal(wh,'x')
  set(gca,'XTick',log(tick),'XTickLabel',ticklabel);
else
  set(gca,'YTick',log(tick),'YTickLabel',ticklabel);
end
