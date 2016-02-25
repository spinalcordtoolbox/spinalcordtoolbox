function m = reshape2D_undo(f,dim,msize)

% function m = reshape2D_undo(f,dim,msize)
%
% <f> has the same dimensions as the output of reshape2D
% <dim> was the dimension of <m> that was used in reshape2D
% <msize> was the size of <m>
%
% return <f> but with the same dimensions as passed to reshape2D.
%
% example:
% a = randn(3,4,5);
% b = reshape2D(a,2);
% isequal(size(b),[4 15])
% c = reshape2D_undo(b,2,size(a));
% isequal(size(c),[3 4 5])
% isequal(a,c)

% figure out the permutation order that was used in reshape2D
dimorder = [dim setdiff(1:max(length(msize),dim),dim)];

% figure out the unsquished size
if dim > length(msize)  % if weird case (the dimension that was shifted was off the deep end), then handle directly
  reshapesize = [1 msize];
else  % otherwise, handle normally
  reshapesize = msize(dimorder);
end

% unsquish and the permute back to the original order
m = ipermute(reshape(f,reshapesize),dimorder);
