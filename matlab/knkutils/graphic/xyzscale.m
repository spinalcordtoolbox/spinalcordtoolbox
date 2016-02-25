function f = xyzscale(v)

% function f = xyzscale(v)
%
% <v> is a scalar or a 3-element vector
%
% return the 3D transformation matrix that achieves
% scaling by <v>.

if length(v)==1
  v = repmat(v,[1 3]);
end

f = [v(1)    0    0 0 
     0    v(2)    0 0 
     0       0 v(3) 0
     0       0    0 1 ];
