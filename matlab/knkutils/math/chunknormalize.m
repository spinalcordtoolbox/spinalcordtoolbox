function f = chunknormalize(m,num)

% function f = chunknormalize(m,num)
%
% <m> is a matrix
% <num> is the number per chunk.  special case is [] which means do nothing.
%
% return <m> except that the nanmean of each successive chunk has been subtracted.
% the dimension of the result is the same as <m>.  the number of elements in <m>
% should be evenly divisible by <num>.
%
% example:
% isequal(chunknormalize([1 2 3 4],2),[-.5 .5 -.5 .5])

if isempty(num)
  f = m;
else
  msize = size(m);
  f = reshape(m(:),num,[]);
  f = bsxfun(@minus,f,nanmean(f,1));
  f = reshape(f,msize);
end
