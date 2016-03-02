function f = drawbars(res,widthfrac,widthfctr,widthmin,lengthfctr,numor,color,bg)

% function f = drawbars(res,widthfrac,widthfctr,widthmin,lengthfctr,numor,color,bg)
%
% <res> is the number of pixels along one side
% <widthfrac> is the fraction of one side for bar width to start at
% <widthfctr> is the scale factor to reduce bar width
% <widthmin> is the minimum allowable bar width in pixels
% <lengthfctr> is how many times longer the length is than the width.
%   0 is a special case which means to use bars that always extend
%   beyond the field-of-view (thus, analogous to lines).
% <numor> is number of orientations (starting at 0 deg)
% <color> is a 3-element vector with the bar color
% <bg> is a 3-element vector with the background color.  [] means do not draw the background.
%
% return a series of 2D images where values are in [0,1].
% the dimensions of the returned matrix are res x res x or*wd
% note that we explicitly convert to grayscale.
%
% example:
% figure; imagesc(makeimagestack(drawbars(32,1/4,2,3,2,4,[0 0 0],[1 1 1]))); axis equal tight;

% NOTE: see also drawcheckerboards.m.

% figure out orientations
ors = linspacecircular(0,pi,numor);

% figure out widths
wds = [];
x0 = 1;
while 1
  if x0*widthfrac*res >= widthmin
    wds = [wds x0];
    x0 = x0/widthfctr;
  else
    break;
  end
end
numwd = length(wds);

% do it
fig = figure;
f = zeros(res,res,numor*numwd);
for p=1:numwd
  for q=1:numor
    clf; drawbar(0,0,0,ors(q),wds(p)*widthfrac,choose(lengthfctr==0,sqrt(2)*1.05,lengthfctr*wds(p)*widthfrac),color,bg);
    f(:,:,(p-1)*numor+q) = renderfigure(res,1);
  end
end  
close(fig);
