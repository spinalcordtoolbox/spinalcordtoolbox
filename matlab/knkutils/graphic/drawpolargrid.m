function f = drawpolargrid(res,numspokes,maxecc,numrings,thickness,color,adjustment)

% function f = drawpolargrid(res,numspokes,maxecc,numrings,thickness,color,adjustment)
%
% <res> is the number of pixels along one side
% <numspokes> is the positive number of spokes starting from 0 degrees (x+ axis)
% <maxecc> is the number of pixels in the radius of the largest ring
% <numrings> is
%    A, the positive number of rings
%   {B} where B is a vector of fractions in [0,1] indicating the ring locations
%       relative to <maxecc>
% <thickness> is the line thickness in points
% <color> is a 3-element vector with the line color
% <adjustment> (optional) is a positive or negative number indicating the number
%   of pixels to add to each ring location.  default: 0.
%
% draw a polar grid consisting of lines that form rings and spokes.
% the spacing of the rings scales with eccentricity (see code for details),
% unless <numrings> is of the {B} case, in which the user directly specifies 
% the ring locations.  return a 2D image where values are in [0,1].
%
% history:
% - 2013/08/18 - add <adjustment> input and special case of <numrings>
%
% example:
% figure; image(drawpolargrid(600,8,300,5,3,[1 0 0])); axis equal tight;

% internal constants
slope = 1/3;

% input
if ~exist('adjustment','var') || isempty(adjustment)
  adjustment = 0;
end

% calc
angs = linspacecircular(0,pi,numspokes);

% adjust
maxecctouse = max(0,maxecc+adjustment);

% draw spokes
fig = figure; hold on;
for p=1:length(angs)
  xmax = cos(angs(p)) * .5 * (2*maxecctouse/res);
  ymax = sin(angs(p)) * .5 * (2*maxecctouse/res);
  h = plot([-1 1] * xmax,[-1 1] * ymax,'r-');
  set(h,'Color',color,'LineWidth',thickness);
end

% figure out ring locations
if iscell(numrings)
  eccs = numrings{1} * maxecc;
else
  options = optimset('Display','off','MaxFunEvals',Inf,'MaxIter',Inf,'TolFun',1e-10,'TolX',1e-10);
  spaceparams = lsqnonlin(@(x) spatialscaling(x,slope,maxecc), ...
    rand(1,numrings-1),zeros(1,numrings-1),maxecc*ones(1,numrings-1),options);
  eccs = [spaceparams maxecc];  % radii in pixels
end

% adjust
eccstouse = max(0,eccs+adjustment);

% draw rings
for p=1:length(eccs)
  h = drawellipse(0,0,0,eccstouse(p)/res,eccstouse(p)/res);
  set(h,'Color',color,'LineWidth',thickness);
end

% finish up
axis([-.5 .5 -.5 .5]);
f = renderfigure(res,2);
close(fig);

%%%%%%%%%%%%%%%%%%%%% HELPER FUNCTION

function [f,ecc,width] = spatialscaling(params,slope,maxecc)

ss = sort([0 params maxecc]);
ecc = (ss(2:end)+ss(1:end-1))/2;
width = ss(2:end)-ss(1:end-1);
f = (slope*ecc - width) ./ (ecc.^.5);
