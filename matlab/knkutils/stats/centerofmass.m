function f = centerofmass(m,dims,dim)

% function f = centerofmass(m,dims,dim)
%
% <m> is a matrix of non-negative values
% <dims> (optional) is a vector of dimensions that the center of mass is to
%   be calculated over.  must consist of one or more unique positive integers.
%   default: 1:ndims(m).
% <dim> (optional) is the dimension along which to concatenate the different
%   center-of-mass coordinates.  default: 1.
% 
% return the matrix coordinates of the center of mass of <m>.
% the coordinates may be in decimals.  the results for the different
% coordinates are concatenated, in the order specified by <dims>, 
% along the dimension <dim>.
%
% example:
% x = abs(randn(100,100,2));
% x(10:20,40:50,:) = 100;
% f = centerofmass(x,[1 2],2);
% figure; imagesc(x(:,:,1)); title(sprintf('center of mass is %s',mat2str(f(1,:,1))));
% figure; imagesc(x(:,:,2)); title(sprintf('center of mass is %s',mat2str(f(1,:,2))));

% input
if ~exist('dims','var') || isempty(dims)
  dims = 1:ndims(m);
end
if ~exist('dim','var') || isempty(dim)
  dim = 1;
end

% sanity check
if any(m(:)<0)
  error('negative values in <m> are not allowed.');
end

% calc
tot = m;
for p=1:length(dims)
  tot = sum(tot,dims(p));
end

% do it
f = [];
for p=1:length(dims)
  tot2 = m .* fillmatrix(1:size(m,dims(p)),size(m),dims(p));
  for q=1:length(dims)
    tot2 = sum(tot2,dims(q));
  end
  f = cat(dim,f,tot2 ./ tot);
end
