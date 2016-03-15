function dist = bootstrapdim(m,dim,fun,num,sz,dim2)

% function dist = bootstrapdim(m,dim,fun,num,sz,dim2)
%
% <m> is a matrix
% <dim> is the dimension along which to draw bootstrap samples
% <fun> is a function that takes a bootstrap sample and outputs a matrix
% <num> (optional) is number of bootstraps to take.  default: 100.
% <sz> (optional) is the size of each bootstrap sample.  default: size(m,dim).
%   if -X, use round(X*size(m,dim)).
% <dim2> (optional) is the dimension to concatenate results along.  default: <dim>.
%
% draw bootstrap samples and apply <fun>.
% concatenate results along <dim2>.
%
% we use parfor to speed things up.
%
% example:
% std(bootstrapdim(randn(1,1000),2,@mean)) < 2*1/sqrt(1000)

% NOTE: some overlap with bootstrp.m and bootstrap.m

% input
if ~exist('num','var') || isempty(num)
  num = 100;
end
if ~exist('sz','var') || isempty(sz)
  sz = size(m,dim);
end
if ~exist('dim2','var') || isempty(dim2)
  dim2 = dim;
end

% calc
len = size(m,dim);
if sz < 0
  sz = round(-sz*len);
end

% do it
dist = {};
parfor p=1:num
  ix = repmat({':'},1,max(ndims(m),dim));
  ix{dim} = ceil(rand(1,sz) * len);
  dist{p} = feval(fun,subscript(m,ix));
end
dist = catcell(dim2,dist);



%%%fprintf('bootstrapdim');
%%%  statusdots(p,num);
%%%fprintf('done.\n');
