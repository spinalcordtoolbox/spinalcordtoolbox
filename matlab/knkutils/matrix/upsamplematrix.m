function f = upsamplematrix(m,n,dim,flag,elt)

% function f = upsamplematrix(m,n,dim,flag,elt)
%
% <m> is a matrix
% <n> is a positive integer with the scale factor.
%   can also be a vector of positive integers indicating
%   scale factors for different dimensions of <m> (ones
%   are automatically assumed for omitted dimensions).
% <dim> should be [] when <n> has more than one element.
%   otherwise, <dim> is the dimension to work upon.
% <flag> (optional) is whether to stop at last element such that
%   the last element of <m> becomes the last element of <f>.
%   default: 0.
% <elt> (optional) is
%   N means fill with N
%   'nearest' means use nearest-neighbor interpolation.
%     in this case, <flag> must be 0.  also, in this case,
%     <n> does not have to be restricted to integers.
%   default: 0.
%
% upsample <m>.
%
% example:
% isequal(upsamplematrix([1 2],3,2),[1 0 0 2 0 0])
% isequal(upsamplematrix([1 2],3,2,1),[1 0 0 2])
% isequal(upsamplematrix([1 2],3,2,[],'nearest'),[1 1 1 2 2 2])
% isequal(upsamplematrix([1 2],[2 2],[],[],'nearest'),[1 1 2 2; 1 1 2 2])

% input
if ~exist('flag','var') || isempty(flag)
  flag = 0;
end
if ~exist('elt','var') || isempty(elt)
  elt = 0;
end

% convert
if length(n)==1
  n = copymatrix(ones(1,ndims(m)),dim,n);
else
  n = placematrix(ones(1,ndims(m)),n,[1 1]);
end

if isequal(elt,'nearest')

  % sanity
  assert(flag==0);

  % determine indices
  idx = {};
  for p=1:ndims(m)
    idx{p} = round(resamplingindices(1,size(m,p),-n(p)));
  end
  
  % do it
  f = m(idx{:});

else

  % calculate desired size
  dsize = zeros(1,ndims(m));
  for p=1:ndims(m)
    dsize(p) = choose(flag,size(m,p)*n(p)-(n(p)-1),size(m,p)*n(p));
  end

  % calculate indices into the new matrix
  ix = repmat({':'},[1 ndims(m)]);
  for p=1:ndims(m)
    ix{p} = 1:n(p):dsize(p);
  end
  
  % do it
  f = repmat(elt,dsize);
  f(ix{:}) = m;

end
