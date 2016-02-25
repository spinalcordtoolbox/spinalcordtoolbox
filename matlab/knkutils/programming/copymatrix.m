function m = copymatrix(varargin)

% function m = copymatrix(m,sub,val)
%
% <m> is a matrix.  can be [] when <sub> is a logical, in which case
%   we assume <m> is zeros(size(<sub>)).
% <sub> is
%   (1) some sort of index (e.g. vector of indices, logical mask)
%   (2) a function that accepts <m> and outputs an index
% <val> is something that can be assigned to the <sub> indices
%
% return a copy of <m> that has <val> shoved into <sub>.
% this function is useful for making modifications of a matrix on-the-fly.
%
% example:
% imagesc(copymatrix(randn(10,10),rand(10,10)>.5,0));
% isequal(copymatrix([1 2 3],@(x) x > 1,1),[1 1 1])
%
% OR
%
% function m = copymatrix(m,sub,dim,val)
%
% <m> is a matrix
% <sub> is a vector of indices or a vector of logicals
% <dim> is a dimension of <m>
% <val> is something that can be assigned to the <sub> indices of <m>,
%   assuming ':' for all other dimensions
%
% return a copy of <m> that has <val> shoved into <sub>.
% this function is useful for making modifications of a matrix on-the-fly.
%
% example:
% copymatrix([1 2 3; 4 5 6],2,1,[1 2 3])

if nargin==3
  m = varargin{1};
  sub = varargin{2};
  val = varargin{3};
  if isa(sub,'logical') && isempty(m)
    m = zeros(size(sub));
  end
  if isa(sub,'function_handle')
    m(feval(sub,m)) = val;
  else
    m(sub) = val;
  end
else
  m = varargin{1};
  sub = varargin{2};
  dim = varargin{3};
  val = varargin{4};
  ix = repmat({':'},[1 max(ndims(m),dim)]);
  ix{dim} = sub;
  m(ix{:}) = val;
end
