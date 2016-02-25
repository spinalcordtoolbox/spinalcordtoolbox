function f = createrescalingmatrix(v)

% function f = createrescalingmatrix(v)
%
% <v> is a scalar or a 3-element vector
%
% return the 3D transformation matrix that achieves
% scaling by <v>, anchoring at (0.5,0.5,0.5).
% this function is useful for the case where 
% the coordinate space is matrix space.

f = xyztranslate(0.5)*xyzscale(v)*xyztranslate(-0.5);
