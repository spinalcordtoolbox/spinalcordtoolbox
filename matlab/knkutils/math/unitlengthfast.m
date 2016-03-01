function [v,len] = unitlengthfast(v,dim)

% function [v,len] = unitlengthfast(v,dim)
%
% <v> is a vector (row or column) or a 2D matrix
% <dim> (optional) is dimension along which vectors are oriented.
%   if not supplied, assume that <v> is a row or column vector.
%
% unit-length normalize <v>.  aside from input flexibility,
% the difference between this function and unitlength.m is that
% we do not deal with NaNs (i.e. we assume <v> does not have NaNs),
% and if a vector has 0 length, it becomes all NaNs.
%
% we also return <len> which is the original vector length of <v>.
% when <dim> is not supplied, <len> is a scalar.  when <dim> is
% supplied, <len> is the same dimensions as <v> except collapsed
% along <dim>.
%
% note some weird cases:
%   unitlengthfast([]) is [].
%   unitlengthfast([0 0]) is [NaN NaN].
%
% example:
% a = [3 0];
% isequalwithequalnans(unitlengthfast(a),[1 0])

if nargin==1
  len = sqrt(v(:).'*v(:));
  v = v / len;
else
  if dim==1
    len = sqrt(sum(v.^2,1));
    v = v ./ repmat(len,[size(v,1) 1]);  % like this for speed.  maybe use the indexing trick to speed up even more??
  else
    len = sqrt(sum(v.^2,2));
    v = v ./ repmat(len,[1 size(v,2)]);
  end
end
