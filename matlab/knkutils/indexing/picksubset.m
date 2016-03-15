function [f,idx,fnot] = picksubset(m,num,seed)

% function [f,idx,fnot] = picksubset(m,num,seed)
%
% <m> is a matrix
% <num> is
%   X indicating the size of the subset to pick out
%   [X Y] where X is the total number of sets and Y is the set number to pull out.
%     in this case, the sets are mutually exclusive and they collectively
%     comprise the original <m> matrix.  note that the number of things in each
%     set may be slightly different.
% <seed> (optional) is the rand state to use.
%   default: 0.
%
% return:
%  <f> as a vector with a random subset of <m>.
%  <idx> as a vector of the indices of the elements that we picked.
%  <fnot> as a vector with the remaining elements of <m>.
%
% note that if you try to pick out a subset bigger than <m>,
% we will just return as many elements as there are in <m>.
%
% example:
% picksubset(randn(10,10),10)

% input
if ~exist('seed','var') || isempty(seed)
  seed = 0;
end

% do it
  prev = rand('state');
  rand('state',seed);
len = length(m(:));
if isscalar(num)
  idx = subscript(randperm(len),1:min(num,len));
else
  numsets = num(1);
  whichset = num(2);
  indices = zeros(numsets,ceil(len/numsets));
  indices(1:len) = randperm(len);
  idx = filterout(indices(whichset,:),0);
end
f = m(idx);
fnot = m(setdiff(1:numel(m),idx));
  rand('state',prev);
