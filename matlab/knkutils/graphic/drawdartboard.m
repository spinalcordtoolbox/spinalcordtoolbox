function eccs = drawdartboard(res,numwedges,numrings,maxecc,polarity,mixfrac,color1,color2,bg,granularity);

% function eccs = drawdartboard(res,numwedges,numrings,maxecc,polarity,mixfrac,color1,color2,bg,granularity);
%
% <res> is
%   0 means standard figure coordinate frame
%  -1 means y-reversed coordinate frame
% <numwedges> is the positive number of wedges in one revolution
% <numrings> is the positive number of rings.  note that the inner radius 
%   of the first ring is 0, so the first ring is really a disc.
% <maxecc> is a number in [0,1].
%   1 means all the way from the center to the edge of the image.
%   <maxecc> indicates the location of the outer edge of the last ring.
% <polarity> is 0 or 1 indicating the polarity of the current case
% <mixfrac> is a fraction in [0,1] indicating how much to mix
%   the two-color checkerboard pattern with a random-color checkerboard
%   pattern.  1 means only use the two-color pattern; 0 means only use the
%   random-color pattern.
% <color1> is a 3-element vector with one color (for the two-color case)
% <color2> is a 3-element vector with the other color (for the two-color case)
% <bg> is a 3-element vector with the background color.
%   [] means do not draw the background.
% <granularity> (optional) is how many points in a complete revolution.  default: 360.
%
% draw a dartboard on the current figure.  we automatically set the
% axis bounds and also reverse the y-axis if necessary.
% return a vector with the radii that bound all of the rings.
%
% notes:
% - the wedges are coincident with the x+ axis.
% - the rings are designed to be scaled with eccentricity (specifically, the relationship
% between the eccentricity of a ring and its width is roughly linear and passes close
% to the origin).
% - for the random-color pattern, the colors are chosen uniformly in RGB space.
% - mixing of colors occurs linearly in RGB space.
%
% example:
% figure; drawdartboard(0,8,3,1,0,.75,[0 0 0],[1 1 1],[.5 .5 .5]); axis equal tight;

% internal constants
slope = 1/3;

% input
if ~exist('granularity','var') || isempty(granularity)
  granularity = 360;
end

% figure out wedge locations
angs = linspace(0,2*pi,numwedges+1);

% figure out ring locations
options = optimset('Display','off','MaxFunEvals',Inf,'MaxIter',Inf,'TolFun',1e-10,'TolX',1e-10);
spaceparams = lsqnonlin(@(x) spatialscaling(x,slope,maxecc), ...
  rand(1,numrings-1),zeros(1,numrings-1),maxecc*ones(1,numrings-1),options);
eccs = sort([0 spaceparams maxecc]);  % all radii including the first and last

% do it
isfirst = 1;
for pp=1:numrings
  for qq=1:numwedges
    theta1 = angs(qq);
    theta2 = angs(qq+1);
    r1 = eccs(pp);
    r2 = eccs(pp+1);
    if isfirst
      bg0 = bg;
      isfirst = 0;
    else
      bg0 = [];
    end
    color = choose(mod(pp+qq+polarity,2)==0,color1,color2);
    color = mixfrac*color + (1-mixfrac)*rand(1,3);
    drawsector(res,theta1,theta2,r1,r2,color,bg0,granularity);
  end
end

%%%%%%%%%%%%%%%%%%%%% HELPER FUNCTION

function [f,ecc,width] = spatialscaling(params,slope,maxecc)

ss = sort([0 params maxecc]);
ecc = (ss(2:end)+ss(1:end-1))/2;
width = ss(2:end)-ss(1:end-1);
f = (slope*ecc - width) ./ (ecc.^.5);
