function [c,acts,result,density] = fitrectpdf(data,ks,valsize)

% function [c,acts,result,density] = fitrectpdf(data,ks,valsize)
%
% <data> is points x dimensions
% <ks> is 1 x K with k-values to try
% <valsize> (optional) is
%   A where 0<A<1 means use fraction A for validation.  in this case, we randomly select.
%   B means use exactly B data points for validation.  in this case, we use the first B points.
%   default: 0.2.
%
% use k-means (cosine distance) to find centers, interpret these as unit-length directions
%   along which we rectify and then square, and then calculate likelihood 
%   of the data (using cross-validation).  note that our likelihood is
%   unnormalized and makes sense only if the entries in <data> are all
%   length-normalized.  also, note that we add a eps offset to ensure that
%   our log probabilities are defined.
% return <c> as k x dimensions with the optimal directions (unit-length).
% return <acts> as points x k with the new activations.
% return <result> as K x 1 with average log probability values for the various cases we tried.
% return <density> as 1 x k vector with number of points assigned to each center.
%
% see also fitrbfpdf.m.
%
% example:
% [c,acts,result,density] = fitrectpdf(randn(1000,2),2:6);
% figure; plot(2:6,result,'ro-'); xlabel('k'); ylabel('avg log prob');

% OBSOLETE:
% % return <result2> as K x 1 with something like <result> except we omit the log2
% result2 = NaN*zeros(length(ks),1);

% constants
expt = 2;

% input
if ~exist('valsize','var') || isempty(valsize)
  valsize = 0.2;
end

% calc
n = size(data,1);
if valsize < 1
  numv = round(valsize*n);
  validx = subscript(randperm(n),1:numv);
else
  assert(isint(valsize));
  numv = valsize;
  validx = 1:numv;
end

% do the main stuff
result = NaN*zeros(length(ks),1);
c = cell(1,length(ks));
density = cell(1,length(ks));
acts = cell(1,length(ks));
for p=1:length(ks)
  k = ks(p);
  fprintf('running kmeans for k=%d.\n',k);
  
  % run k-means
  [idx,c{p}] = runkmeans(data(setdiff(1:n,validx),:),k,'cosine');
  c{p} = unitlength(c{p},2);
  density{p} = countinstances(idx);

  % linear
  result(p) = calcrectlikelihood(c{p},data(validx,:));

end

% return the best one
[mx,ix] = max(result);
c = c{ix};
density = density{ix};
acts = acts{ix};
