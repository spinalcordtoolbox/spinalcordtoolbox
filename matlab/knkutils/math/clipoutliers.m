function [m,pmx,pmn,numhigh,numlow] = clipoutliers(m,subnum,threshs)

% function [m,pmx,pmn,numhigh,numlow] = clipoutliers(m,subnum,threshs)
%
% <m> is a matrix
% <subnum> (optional) is the number of elements of <m> to consider
%   when calculating the percentiles.  default is [] which means
%   to consider all elements.  if supplied, the subset we draw from <m>
%   is random but deterministic (i.e. the same each time).  the point
%   of this input is to increase speed.
% <threshs> (optional) is [A B] where A is a number of percentiles and B is a
%   multiplier.  for example, [25 100] means find the median, add 100 times the 
%   difference between the median and the 75th percentile to get the upper
%   cutoff, and subtract 100 times the difference between the median and the
%   25th percentile to get the lower cutoff.  default: [25 100].
%   to be more aggressive in cutting out outliers, try [40 4].
%
% return <m> but with outliers clipped.
% also return the upper and lower cutoffs in <pmx> and <pmn>.
% also return the number of elements that were above the upper
%   cutoff and below the lower cutoff in <numhigh> and <numlow>.
%
% specifically, what we do is to look at these two cutoffs:
%   P50 + threshs(2)*(P(50+threshs(1))-P50) and 
%   P50 - threshs(2)*(P50-P(50-threshs(1)))
% any values larger than the first cutoff are set to the maximum
% of the remaining values.  any values smaller than the second
% cutoff are set to the minimum of the remaining values.
%
% example:
% x = [randn(1,100) 1e10];
% figure; hist(x);
% figure; hist(clipoutliers(x));

% input
if ~exist('subnum','var') || isempty(subnum)
  subnum = [];
end
if ~exist('threshs','var') || isempty(threshs)
  threshs = [25 100];
end

% do it
if isempty(subnum)
  vals = prctile(m(:),[50-threshs(1) 50 50+threshs(1)]);
else
  vals = prctile(picksubset(m,subnum),[50-threshs(1) 50 50+threshs(1)]);
end
pmx = vals(2) + threshs(2)*(vals(3)-vals(2));
pmn = vals(2) - threshs(2)*(vals(2)-vals(1));
toohigh = m > pmx;
m(toohigh) = max(m(~toohigh));
toolow = m < pmn;
m(toolow) = min(m(~toolow));
numhigh = count(toohigh);
numlow = count(toolow);
