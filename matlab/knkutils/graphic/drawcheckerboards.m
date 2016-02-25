function f = drawcheckerboards(res,widthfrac,widthfctr,widthmin,heightfctr,numor,numphwidth,numphheight,color1,color2)

% function f = drawcheckerboards(res,widthfrac,widthfctr,widthmin,heightfctr,numor,numphwidth,numphheight,color1,color2)
%
% <res> is the number of pixels along one side
% <widthfrac> is the fraction of one side for check width to start at
% <widthfctr> is the scale factor to reduce check width
% <widthmin> is the minimum allowable check width in pixels
% <heightfctr> is how many times longer the height is than the width.
%   0 is a special case which means to use checks that always extend beyond
%   the field-of-view (thus, analogous to bars).  in this case,
%   <numphheight> should be set to 1.
% <numor> is number of orientations (starting at 0 deg).
%   if negative, this means to proceed only to 90 deg (not 180 deg).
%   if imaginary (e.g. 8j), this means to go from -90 to 90 deg.
%     the point of this is to match the conventions of makegratings2d.m
%     (in the sense that a vertical bar gets rotated to be like a
%     horizontal grating).
% <numphwidth> is number of phases in the width direction
% <numphheight> is number of phases in the height direction
% <color1> is a 3-element vector with color of center check
% <color2> is a 3-element vector with the other color
%
% return a series of 2D images where values are in [0,1].
% the dimensions of the returned matrix are res x res x phh*phw*or*wd
% note that we explicitly convert to grayscale.
%
% example:
% figure; imagesc(makeimagestack(drawcheckerboards(64,1/2,2,8,.5,3,3,1,[0 0 0],[1 1 1]))); axis equal tight;

% NOTE: see also drawbars.m.

% figure out orientations
if imag(numor) ~= 0
  numor = imag(numor);
  ors = linspacecircular(-pi/2,pi/2,numor);
else
  if numor > 0
    ors = linspacecircular(0,pi,numor);
  else
    numor = -numor;
    ors = linspacecircular(0,pi/2,numor);
  end
end

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

% figure out phases
phsw = linspacecircular(0,2*pi,numphwidth);
phsh = linspacecircular(0,2*pi,numphheight);

% do it
fig = figure;
f = zeros(res,res,numphheight*numphwidth*numor*numwd);
cnt = 1;
for p=1:numwd
  for q=1:numor
    for r=1:numphwidth
      for s=1:numphheight
        clf; drawcheckerboard(0,phsw(r),phsh(s),ors(q),wds(p)*widthfrac,choose(heightfctr==0,0,heightfctr*wds(p)*widthfrac),color1,color2);
        f(:,:,cnt) = renderfigure(res,1);
        cnt = cnt + 1;
      end
    end
  end
end  
close(fig);
