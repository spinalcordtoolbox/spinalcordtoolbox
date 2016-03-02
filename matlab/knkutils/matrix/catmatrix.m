function f = catmatrix(varargin)

% function f = catmatrix(dim,m1,m2,...)
%
% <dim> is the dimension to concatenate along
% <m1>,<m2>,... are matrices
%
% concatenate <m1>,<m2>,etc. along <dim>.
% matrices are automatically expanded and filled with 0s;
% all matrices are aligned to the "first corner".
%
% example:
% figure; imagesc(catmatrix(2,randn(3,5),randn(5,3),randn(6,6))); axis equal tight;

% calc
dim = varargin{1};

% figure out maximum size [SHOULD BE A SEPARATE FUNCTION?]
ndim = 0; sz = [];
for p=2:length(varargin)
  ndim = max(ndim,ndims(varargin{p}));  % max number of dimensions
  sz0 = sizefull(varargin{p},ndim);     % current matrix's size
  if length(sz0) > length(sz)
    sz = placematrix(ones(1,length(sz0)),sz,[1 1]);  % expand out the current max size
  end
  sz = max([sz; sz0],[],1);  % figure out maximum size so far
end

% expand each matrix as necessary
for p=2:length(varargin)
  varargin{p} = placematrix2(zeros(copymatrix(sz,dim,size(varargin{p},dim))),varargin{p});
end

% return
f = cat(dim,varargin{2:end});
