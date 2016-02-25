function [f,perm] = permutedim(m,dim,perm,wantind)

% function [f,perm] = permutedim(m,dim,perm,wantind)
%
% <m> is a matrix
% <dim> (optional) is a dimension of <m>.  special case is 0 which means permute globally.
%   another special case is -1 which means permute globally and ensure that the
%   original order is NOT obtained (however, if <perm> is supplied, that takes
%   precedence).  default: 0.
% <perm> (optional) is the permutation order to use
% <wantind> (optional) is whether you want individual cases
%   permuted randomly.  default: 0.  note that when <dim> is 0 or -1,
%   <wantind> has no meaning.
%
% randomly shuffle <m> along <dim> or globally.
% also return <perm>, the permutation order used.
%
% example:
% a = repmat(1:9,[5 1])
% b = permutedim(a,2)
% b2 = permutedim(a,2,[],1)
% b3 = permutedim(a,0)

% deal with input
if ~exist('dim','var') || isempty(dim)
  dim = 0;
end
if ~exist('perm','var') || isempty(perm)
  perm = [];
end
if ~exist('wantind','var') || isempty(wantind)
  wantind = 0;
end

% do it
if dim==0 || dim==-1

  % figure out perm
  if isempty(perm)
    while 1
      perm = randperm(length(m(:)));
      if dim==-1
        if ~isequal(perm,1:length(m(:)))
          break;
        end
      else
        break;
      end
    end
  end
  
  % do it
  f = reshape(m(perm),size(m));

else
  if wantind
  
    % make 2D
    f = reshape2D(m,dim);
    
    % figure out perm
    if isempty(perm)
      [d,perm] = sort(rand(size(f,1),size(f,2)));  % figure out random permutation for each column
      perm = perm + repmat((0:size(f,2)-1)*size(f,1),[size(f,1) 1]);  % add appropriate offsets
    end
    
    % index into f and then undo the 2D
    f = reshape2D_undo(f(perm),dim,size(m));
  
  else
  
    % figure out perm
    if isempty(perm)
      perm = randperm(size(m,dim));
    end
  
    % construct indices
    indices = repmat({':'},[1 max(ndims(m),dim)]);
    indices{dim} = perm;
    
    % do it
    f = subscript(m,indices);
  
  end
end
