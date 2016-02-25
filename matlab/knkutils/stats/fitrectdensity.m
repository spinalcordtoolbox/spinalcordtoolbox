function [c,density,meancorr,respcorr,sumact,err,meansparse,acts] = fitrectdensity(data,k,valsize,maxiters,iterfun)

% function [c,density,meancorr,respcorr,sumact,err,meansparse,acts] = fitrectdensity(data,k,valsize,maxiters,iterfun)
%
% <data> is points x dimensions.  each row should be unit-length.
% <k> is the k-value to try.  special case is a matrix of dimensions centers x dimensions.
%   in this case, we do not run k-means and interpret <k> as the unit-length directions.
% <valsize> (optional) is
%   A where 0<A<1 means use fraction A for validation.  in this case, we randomly select.
%   B means use exactly B data points for validation.  in this case, we use the first B points.
%   default: 0.2.
% <maxiters> (optional) is the input to pass to runkmeans.m.
% <iterfun> (optional) is the input to pass to runkmeans.m.
%
% use k-means (cosine distance) to find centers and interpret these as unit-length directions
%   along which we positive-rectify and then square.
% return <c> as k x dimensions with the unit-length directions.
% return <density> as 1 x k vector with number of validation points assigned to each center.
% return <meancorr> as the mean across validation points of the dot product with the nearest center.
% return <respcorr> as the mean second-best correlation of half-sq responses (using whole data set).
% return <sumact> as points x 1 with the total activity (sum across half-sq responses).
% return <err> as 1 x iterations with the errors from runkmeans.m.
% return <meansparse> as the mean channel sparseness.
% return <acts> as points x channels with the half-squared activities.
%
% based loosely on fitrectpdf.m.
%
% example:
% data = unitlength(randn(1000,2),2);
% meancorr = []; respcorr = [];
% for k=2:20
%   [c,density,meancorr(k-1),respcorr(k-1),sumact] = fitrectdensity(data,k,100);
% end
% figure; plot(2:20,meancorr,'ro-'); xlabel('k'); ylabel('meancorr');
% figure; plot(2:20,respcorr,'ro-'); xlabel('k'); ylabel('respcorr');

% constants
expt = 2;

% inputs
if ~exist('valsize','var') || isempty(valsize)
  valsize = 0.2;
end
if ~exist('maxiters','var') || isempty(maxiters)
  maxiters = [];
end
if ~exist('iterfun','var') || isempty(iterfun)
  iterfun = [];
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

% run k-means
if size(k,2) > 1
  c = k;
  err = [];
else
  fprintf('running kmeans for k=%d.\n',k);
  [idx,c,err] = runkmeans(data(setdiff(1:n,validx),:),k,'cosine',[],maxiters,iterfun);
end
c = unitlength(c,2);

% for each val point, measure dist (dot product) to nearest center.  take the mean across val points.
[mx,ix] = max(data(validx,:)*c',[],2);
density = countinstances(ix);
meancorr = mean(mx);

% calculate half-sq activity
acts = posrect(data*c').^expt;

% calculate average second-best correlation of responses
cm = sort(calcconfusionmatrix(acts),2);
if size(cm,2)==1
  respcorr = mean(cm(:,end));
else
  respcorr = mean(cm(:,end-1));
end

% calculate summed activity for each data point
sumact = sum(acts,2);

% calculate mean sparseness for each channel
meansparse = mean(calcsparseness(acts,1));
