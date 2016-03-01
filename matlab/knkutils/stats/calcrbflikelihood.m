function f = calcrbflikelihood(varargin)

% function f = calcrbflikelihood(c,bs,data)
%
% <c> is rbfnumber x dimensions with the centers of the RBFs
% <bs> is 1 x B with b-values to use
% <data> is points x dimensions with points to evaluate the likelihood at
%
% OR
%
% function f = calcrbflikelihood(dist,bs,numdims,flag)
%
% <dist> is rbfnumber x points with the squared distance of each RBF to each point
% <bs> is 1 x B with b-values to use
% <numdims> is the number of dimensions that the RBFs are defined on
% <flag> should be passed in as 1
%
% calculate the likelihood of a set of points given a collection of RBFs.
% we assume RBFs have the same bandwidth and same gain.
% return a 1 x B vector of average log probability values.
%
% see also fitrbfpdf.m.
%
% example:
% calcrbflikelihood(randn(10,50),[2 4],randn(1000,50))

% the first case
if length(varargin) < 4

  c = varargin{1};
  bs = varargin{2};
  data = varargin{3};

  % calc
  numrbf = size(c,1);
  numdims = size(c,2);
  numpoints = size(data,1);
  
  % calculate squared distance between each center and each point 
  dist = zeros(numrbf,numpoints);
  for zz=1:numpoints
    dist(:,zz) = sum(bsxfun(@minus,data(zz,:),c).^2,2);
  end

% the second case
else

  dist = varargin{1};
  bs = varargin{2};
  numdims = varargin{3};
  
  % calc
  numrbf = size(dist,1);
  numpoints = size(dist,2);

end

% evaluate likelihood using different bandwidths
f = zeros(1,length(bs));
for zz=1:length(bs)
  b = bs(zz);
  s = sqrt(1/(2*b));
  aa = log2(1/numrbf) + numdims * log2(1/(s*sqrt(2*pi)));
  bb = log2(sum(exp(-b*dist),1));
  f(zz) = aa + mean(bb);  % this is log probability divided by number of samples
end
