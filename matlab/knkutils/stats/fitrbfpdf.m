function [c,b,acts,result,density] = fitrbfpdf(data,ks,bs,valsize)

% function [c,b,acts,result,density] = fitrbfpdf(data,ks,bs,valsize)
%
% <data> is points x dimensions
% <ks> is 1 x K with k-values to try
% <bs> is 1 x B with b-values to try
% <valsize> (optional) is
%   A where 0<A<1 means use fraction A for validation
%   B means use exactly B data points for validation
%   default: 0.2.
%
% use k-means to find RBF centers and then optimize bandwidth for maximum
%   likelihood of the data (using cross-validation).
% return <c> as k x dimensions with the optimal centers.
% return <b> with the optimal b value.
% return <acts> as points x k with the new RBF activations.
% return <result> as K x B with average log probability values for the various cases we tried.
% return <density> as 1 x k vector with number of points assigned to each center (in the optimal case).
%
% see also calcrbflikelihood.m.
%
% example:
% data = [randn(1000,2); randn(1000,2)+3];
% [c,b,acts,result,density] = fitrbfpdf(data,1:3,.2:.2:2);
% figure; imagesc(result); xlabel('b'); ylabel('k');
% for p=1:size(acts,2)
%   figure; hold on;
%   scatter(data(:,1),data(:,2),40,acts(:,p));
% end

% input
if ~exist('valsize','var') || isempty(valsize)
  valsize = 0.2;
end

% calc
n = size(data,1);
if valsize < 1
  numv = round(valsize*n);
else
  assert(isint(valsize));
  numv = valsize;
end

% figure out validation set
order = randperm(n);

% do the main stuff
result = NaN*zeros(length(ks),length(bs));
c = cell(1,length(ks));
density = cell(1,length(ks));
for p=1:length(ks)
  k = ks(p);
  fprintf('running kmeans for k=%d.\n',k);
  
  % run k-means
  [idx,c{p}] = runkmeans(data(order(1:end-numv),:),k);
  density{p} = countinstances(idx);

  % figure out bandwidth
  result(p,:) = calcrbflikelihood(c{p},bs,data(order(end-numv+1:end),:));

end

% which was best model?
[mx,ix] = max(result(:));

% return the best one
c = c{mod2(ix,length(ks))};
density = density{mod2(ix,length(ks))};
b = bs(ceil(ix/length(ks)));
k = size(c,1);

% calculate new activations
acts = NaN*zeros(n,k);
for p=1:k
  acts(:,p) = exp(-b*sum(bsxfun(@minus,data,c(p,:)).^2,2));
end

