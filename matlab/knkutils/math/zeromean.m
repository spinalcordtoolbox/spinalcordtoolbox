function f = zeromean(m,dim)

% function f = zeromean(m,dim)
%
% <m> is a matrix
% <dim> (optional) is the dimension of interest.
%   if supplied, subtract off the mean of each case oriented along <dim>.
%   if [] or not supplied, subtract off the global mean.
%
% subtract off the mean of <m>, either of individual cases or the global mean.
% we use nanmean to deal with NaNs gracefully.
%
% note some weird cases:
%   zeromean([]) is []
%   zeromean([NaN NaN]) is [NaN NaN]
%
% example:
% a = [1 1; 2 0];
% isequal(zeromean(a),[0 0; 1 -1])
% a = [1 NaN];
% isequalwithequalnans(zeromean(a,1),[0 NaN])

% do it
if ~exist('dim','var') || isempty(dim)
  f = m - nanmean(m(:));
else
  f = bsxfun(@minus,m,nanmean(m,dim));
end
