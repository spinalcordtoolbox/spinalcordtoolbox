function f = plusmulti(varargin)

% function f = plusmulti(varargin)
%
% <varargin> are zero or more matrices of the same dimensions
%
% return the sum of all the matrices.  the usefulness of this
% function is to avoid having to explicitly sum the arguments 
% (e.g. a+b) and to avoid having to concatenate the arguments in a 
% matrix (e.g. sum(cat(1,x{:}),1)).
% 
% example:
% isequal(plusmulti(1,2,3),6)

f = [];
for p=1:nargin
  if p==1
    f = varargin{p};
  else
    f = f + varargin{p};
  end
end
