function f = rowfun(x,fun,dim)

% function f = rowfun(x,fun,dim)
%
% <x> is a 2D matrix
% <fun> is a function that accepts a row and outputs a matrix
% <dim> (optional) is the dimension along which to concatenate results.
%   default: 1.
%
% apply <fun> to each row of <x> and concatenate the results.
% for speed reasons, we assume that each of the various results
%   has the same dimensions.
%
% example:
% isequal(rowfun([1 1; 2 2],@(x) sum(x)),[2 4]')

% input
if ~exist('dim','var') || isempty(dim)
  dim = 1;
end

% get out early
if isempty(x)
  f = [];
  return;
end

% prep
temp = feval(fun,x(1,:));  % do the first one
nn = size(temp,dim);       % how big is each one?
dsize = size(temp);
dsize(dim) = dsize(dim)*size(x,1);  % assume each result is same size!

% do it
f = zeros(dsize);
ix = repmat({':'},[1 length(dsize)]);
for p=1:size(x,1)
  ix{dim} = (p-1)*nn + (1:nn);
  if p==1
    f(ix{:}) = temp;
  else
    f(ix{:}) = feval(fun,x(p,:));
  end
end



% OLD SLOW WAY:
% f = [];
% for p=1:size(x,1)
%   f = cat(dim,f,feval(fun,x(p,:)));
% end
