function f = mtimesmulti(varargin)

% function f = mtimesmulti(varargin)
%
% <varargin> are zero or more matrices
%
% return the sequential multiplication of all the matrices.  the usefulness of this
% function is to avoid having to explicitly multiply the arguments (e.g. a*b).
% 
% example:
% isequal(mtimesmulti(1,2,3),6)

f = [];
for p=1:nargin
  if p==1
    f = varargin{p};
  else
    f = f*varargin{p};
  end
end
