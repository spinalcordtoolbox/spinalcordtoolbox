function f = constructfiltertiling(x)

% function f = constructfiltertiling(x)
%
% <x> is a 2D (NxN) square matrix with the base filter.
%   a special case is that <x> can have stuff along the third dimension.
%
% return tiled filters as res x res x X*X where X is N+(N-1).
% portions of filters that are not covered by the base filter
% are set to zero.  see constructfiltersubsample.m.
%
% in the special case, the tiled filters are returned as
% res x res x stuff x X*X.
%
% example:
% figure; imagesc(makeimagestack(constructfiltertiling(randn(4,4))));

n = size(x,1);
f = constructfiltersubsample(n,x,1-ceil((n-1)/2) : 1-ceil((n-1)/2)+(n+(n-1)-1));
