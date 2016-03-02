function [f,xx,yy] = makeconcentricgrating2d(res,cpfov,phase,xx,yy)

% function [f,xx,yy] = makeconcentricgrating2d(res,cpfov,phase,xx,yy)
%
% <res> is the number of pixels along one side
% <cpfov> is the number of cycles per field-of-view
% <phase> is the phase in [0,2*pi)
% <xx>,<yy> (optional) are speed-ups (dependent on <res>)
%
% return an image where values are in [-1,1].
% we don't normalize the matrix for power or anything like that.
%
% example:
% figure; imagesc(makeconcentricgrating2d(100,4,0),[-1 1]);

% construct coordinates
if ~exist('xx','var') || isempty(xx)
  [xx,yy] = calcunitcoordinates(res);
end

% do it
f = cos(2*pi*cpfov*sqrt(xx.^2+yy.^2) + phase);
