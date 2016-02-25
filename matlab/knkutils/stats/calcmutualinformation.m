function [f,n1,n2,je] = calcmutualinformation(m1,m2,numbins)

% function f = calcmutualinformation(m1,m2,numbins)
%
% <m1>,<m2> are matrices of the same size
% <numbins> (optional) is the number of bins to use.  default: 10.
%
% calculate the mutual information between <m1> and <m2>.
% the idea is that we bin the distribution of values in
% <m1> and <m2> separately.  then we calculate the joint
% distribution between <m1> and <m2>.  the mutual information
% is the sum of the log2-entropies of <m1> and <m2> considered
% separately minus the log2-entropy of the joint distribution.
% 
% the idea behind the joint distribution is
% that given that an element maps to a certain bin of <m1>,
% where does that element map to in the bins of <m2>?  if
% <m1> tells us nothing about <m2>, then the mapping to
% the bins of <m2> will be distributed the same as the
% marginal distribution of <m2>.
%
% if <m1> and <m2> are empty, we return NaN.
%
% note that to avoid NaN issues, we remove all pairs in <m1>
% and <m2> that have at least one NaN in them.
%
% example:
% calcmutualinformation(randn(1000,1000),randn(1000,1000)) < .001
%
% history:
% 2010/06/05 - remove NaNs from both <m1> and <m2>.

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

% hack so that we don't get anything in the last degenerate bin
r1(end) = r1(end)+1;
r2(end) = r2(end)+1;

% calculate number in each bin (n) and the index of the bin that each element gets put into (b)
[n1,b1] = histc(m1(:),r1);
[n2,b2] = histc(m2(:),r2);

% calculate joint entropy
je = zeros(numbins,numbins);
idx = (b2-1)*numbins + b1;
for p=1:length(idx)
  je(idx(p)) = je(idx(p)) + 1;  % TODO: is there a faster way?
end

% sanity check
assert(sum(n1)==sum(n2) & sum(n2)==sum(je(:)));

% finally
%fprintf('entropy1: %.5f\nentropy2: %.5f\nje: %.5f\n',calcentropy(n1),calcentropy(n2),calcentropy(je));
f = calcentropy(n1) + calcentropy(n2) - calcentropy(je);
