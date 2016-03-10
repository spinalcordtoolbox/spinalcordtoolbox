function [f,xx,yy] = makegrating2d(res,cpfov,ang,phase,xx,yy,r,c)

% function [f,xx,yy] = makegrating2d(res,cpfov,ang,phase,xx,yy,r,c)
%
% <res> is the number of pixels along one side
% <cpfov> is the number of cycles per field-of-view
% <ang> is the orientation in [0,2*pi).  0 means a horizontal grating.
% <phase> is the phase in [0,2*pi)
% <xx>,<yy> (optional) are speed-ups (dependent on <res>)
% <r> (optional) is the row associated with the center of the grating (can be a decimal).
%   default to the exact center of the image along the vertical dimension.
% <c> (optional) is the column associated with the center of the grating (can be a decimal).
%   default to the exact center of the image along the horizontal dimension.
%
% return an image where values are in [-1,1].
% we don't normalize the matrix for power or anything like that.
%
% example:
% figure; imagesc(makegrating2d(100,4,pi/6,0),[-1 1]);

% construct coordinates
if ~exist('xx','var') || isempty(xx)
  [xx,yy] = calcunitcoordinates(res);
end
if ~exist('r','var') || isempty(r)
  r = (1+res)/2;
end
if ~exist('c','var') || isempty(c)
  c = (1+res)/2;
end

% do it
f = cos(2*pi*cpfov*(-sin(ang)*(xx - (c-(1+res)/2)/res) + cos(ang)*(yy + (r-(1+res)/2)/res)) + phase);
  % the idea is like this.  the "base" case is a horizontal grating
  % that is peaked at the origin.  this can be obtained by simply
  % evaluating cos(2*pi*cpfov*y + phase) where y is the y-coordinate.
  %
  % now, we stipulate that positive orientation means to 
  % rotate the grating CCW.  so, for a given grating, to figure
  % out the values, we can first undo the rotation and then sample
  % from the "base" case.  to undo CCW, we just have to rotate CW,
  % which is like [x' y']' = [cos ang  sin ang; -sin ang  cos ang] [x y]'.
  % since we only care about the y-coordinate, we can substitute
  % -sin ang * x + cos ang * y for Y in the equation above, and we're done.
  %
  % UPDATE: but we also want to be able to easily translate the center of the grating.
  % so, we just tacked on the modifications to xx and yy in the equation.  the sign
  % issues are tricky, so we just flipped the signs until the function did the right thing.
