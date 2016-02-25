function f = chunkfun(v,nums,fun)

% function f = chunkfun(v,nums,fun)
%
% <v> is a row or column vector
% <nums> is a vector of counts such that sum(<nums>) is equal to length(<v>)
% <fun> (optional) is a function that takes A x B to 1 x B.  each column consists
%   of a mix of valid elements (comprising a group in <v>) and NaNs.  <fun>
%   should compute something based on the valid elements in each column.
%   default: @(y)nansum(y,1).
%
% return a row vector by applying <fun> to <v> reshaped into groups.
% we emphasize fast execution!
%
% example:
% isequal(chunkfun([1 2 3 5 6],[3 2]),[6 11])

% input
if ~exist('fun','var') || isempty(fun)
  fun = @(y)nansum(y,1);
end

% do it
f = NaN(max(nums),length(nums));
f(bsxfun(@le,repmat((1:max(nums))',[1 length(nums)]),nums)) = v;
f = feval(fun,f);
