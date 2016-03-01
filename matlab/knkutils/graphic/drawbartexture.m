function f = drawbartexture(res,widthfrac,lengthfrac,ang,n,fov,color,bg)

% function f = drawbartexture(res,widthfrac,lengthfrac,ang,n,fov,color,bg)
%
% <res> is the number of pixels along one side
% <widthfrac> is the fraction of one side for bar width
% <lengthfrac> is the fraction of one side for bar length
% <ang> is the orientation in [0,2*pi).  0 means a horizontal bar.
% <n> is the number of bars to place
% <fov> is the scale factor on the FOV that defines the range
%   over which we randomly choose bar center positions (3 means 
%   3x3 with the actual FOV at the center)
% <color> is a 3-element vector with the bar color
% <bg> is a 3-element vector with the background color.  [] means do not draw the background.
%
% OR, instead of <n> and <fov>, you can have
% ...
% <n> is number of grid points along each side (there is a grid point placed at each corner)
% <noise> is gain on Gaussian noise added to grid positions (note that the field-of-view
%   has length 1).  we detect this case by checking whether <noise> is less than 1.
% ...
%
% return a 2D image where values are in [0,1].
% note that we explicitly convert to grayscale.
%
% example:
% figure; imagesc(drawbartexture(500,1/32,1/8,pi/6,300,2,[0 0 0],[1 1 1])); axis equal tight;
% figure; imagesc(drawbartexture(500,1/64,1/16,pi/4,20,1/60,[0 0 0],[1 1 1])); axis equal tight;

fig = figure;
if fov < 1
  xs = linspace(-.5,.5,n);
  ys = linspace(-.5,.5,n);
  for p=1:n
    for q=1:n
      drawbar(0,xs(p)+fov*randn,ys(q)+fov*randn,ang,widthfrac,lengthfrac,color,choose(p==1 & q==1,bg,[]));
    end
  end
else
  for p=1:n
    drawbar(0,(rand-.5)*fov,(rand-.5)*fov,ang,widthfrac,lengthfrac,color,choose(p==1,bg,[]));
  end
end
f = renderfigure(res,1);
close(fig);
