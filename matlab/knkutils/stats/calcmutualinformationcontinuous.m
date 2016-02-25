function f = calcmutualinformationcontinuous(m1,m2,numbins)

% function f = calcmutualinformationcontinuous(m1,m2,numbins)
%
% <m1>,<m2> are matrices of the same size
% <numbins> (optional) is the number of bins to use.  default: 10.
%
% this function is roughly analogous to calcmutualinformation.m
% except that we use a kernel-based strategy to achieve a mutual
% information metric that is continuous (i.e. does not suffer from
% the discretization that afflicts a histogram-based strategy).
%
% our strategy is as follows:
% - carve up the data space by using <numbins> bins between the
%   minimum and maximum of each of the matrices <m1> and <m2>.
% - find the middle point of each bin; this defines an M x N grid.
% - use kernels to estimate the data density at each point of
%   the grid.  specifically, we use an Epanechnikov kernel whose
%   bandwidth along each dimension is equal to the size of the 
%   bins along that dimension.  each data point can be thought
%   of as contributing one kernel, and the total data density
%   is given by the sum of all the kernels.
% - the data density at the grid points is then treated as
%   the joint distribution between <m1> and <m2>.
% - we calculate the sum of the entropy of the marginals of the joint 
%   distribution along the first dimension and the entropy of the
%   marginals of the joint distribution along the second dimension
%   and then subtract the entropy of the entire joint distribution.
%   this gives us the final mutual information value.
%
% if <m1> and <m2> are empty, we return NaN.
%
% note that to avoid NaN issues, we remove all pairs in <m1>
% and <m2> that have at least one NaN in them.
%
% example:
% calcmutualinformationcontinuous(randn(1000,1000),randn(1000,1000))

% input
if ~exist('numbins','var') || isempty(numbins)
  numbins = 10;
end

% get rid of NaNs
bad = isnan(m1) | isnan(m2);
m1 = m1(~bad);
m2 = m2(~bad);

% sanity
if isempty(m1)
  f = NaN;
  return;
end

% define bin boundaries
r1 = linspace(min(m1(:)),max(m1(:)),numbins+1);
r2 = linspace(min(m2(:)),max(m2(:)),numbins+1);

% calculate midpoints of bins
mid1 = (r1(1:end-1)+r1(2:end))/2;
mid2 = (r2(1:end-1)+r2(2:end))/2;

% calculate the width of each bin
sz1 = (r1(end)-r1(1))/numbins;
sz2 = (r2(end)-r2(1))/numbins;

% calculate joint entropy
allpoints = [flatten(m1); flatten(m2)];
je = zeros(length(mid1),length(mid2));
for p=1:length(mid1)
  for q=1:length(mid2)
    ix = findlocal([mid1(p) mid2(q)],allpoints,[sz1 sz2]);
    temp = (abs(m1(ix)-mid1(p))/sz1).^2 + (abs(m2(ix)-mid2(q))/sz2).^2;  % square of (normalized) distance
    good = find(temp <= 1);
    k = vflatten(0.75*(1-temp(good)));
    je(p,q) = sum(k);
  end
end
%%%figure(456); clf; imagesc(je); axis equal tight; caxis([0 max(je(:))]); colormap(jet); colorbar;

% finish up
f = calcentropy(sum(je,1)) + calcentropy(sum(je,2)) - calcentropy(je);





% if isempty(pct)
% else
%   rng1 = prctile(m1(:),[pct 100-pct]);
%   rng2 = prctile(m2(:),[pct 100-pct]);
%   r1 = linspace(rng1(1),rng1(2),numbins+1);
%   r2 = linspace(rng2(1),rng2(2),numbins+1);
% end
% 
% 
% if ~exist('pct','var') || isempty(pct)
%   pct = [];
% end
% 
% % <pct> (optional) is a percentile like 5.  if supplied, we use
% %   bins that start and end at percentiles <pct> and 100-<pct>.
% %   if [] or not supplied, we start and end at the minimum and
% %   maximum values.  default: [].
% 
%  (or
% %   percentiles of these matrices if <pct> is supplied).

