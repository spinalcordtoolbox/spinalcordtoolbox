function f = splitmatrix(m,dim,splt)

% function f = splitmatrix(m,dim,splt)
%
% <m> is a matrix
% <dim> is a dimension
% <splt> (optional) is a vector of positive integers indicating
%   how to perform the split.  default: ones(1,size(m,dim)).
%   you can also flip the sign of entries to indicate that you
%   do not want that entry returned.  special case is <splt>==0
%   which means use <splt> equal to size(m,dim).
%
% split <m> along dimension <dim>, returning a cell vector of matrices.
%
% example:
% isequal(splitmatrix([1 2; 3 4],2),{[1 3]' [2 4]'})
% isequal(splitmatrix([1 2 3 4],2,[2 -1 1]),{[1 2] [4]})

% input
if ~exist('splt','var') || isempty(splt)
  splt = [];  % deal with later
end
if isequal(splt,0)
  splt = size(m,dim);
end

% what is the max number of dimensions involved?
maxdim = max(ndims(m),dim);                % 5

% figure out the dimensions of m
msize = ones(1,maxdim);
msize(1:ndims(m)) = size(m);               % [50 60 40 1 2]

% convert to cell
msize = num2cell(msize);                   % {50 60 40 1 2}

% hack it in
if isempty(splt)
  splt = ones(1,size(m,dim));
end
msize{dim} = abs(splt);                    % {50 60 40 1 [1 1]}

% do it
  prev = warning('query'); warning('off');
f = flatten(mat2cell(m,msize{:}));
  warning(prev);
f = f(splt > 0);
