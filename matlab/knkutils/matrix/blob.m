function m = blob(m,dim,num)

% function m = blob(m,dim,num)
%
% <m> is a matrix
% <dim> is the dimension to work on
% <num> is the positive number of elements to blob over
% 
% sum over each successive group of size <num> along dimension <dim>.
% we assume that the size of <m> along <dim> is evenly divisible by <num>.
%
% example:
% isequal(blob([1 2 3 4; 1 1 1 1],2,2),[3 7; 2 2])

% get out early
if dim > ndims(m)
  assert(num==1);
  return;
end

% desired size of the result
dsize = size(m);
dsize(dim) = dsize(dim)/num; assert(isint(dsize(dim)));

% do it
m = reshape2D(m,dim);
m = reshape2D_undo(squish(sum(reshape(m,num,size(m,1)/num,[]),1),2),dim,dsize);
