function dist = bootstrap(v,fun,num,sz)

% function dist = bootstrap(v,fun,num,sz)
%
% <v> is a row or column vector
% <fun> is a function that takes a row vector and outputs a scalar
% <num> (optional) is number of bootstraps to take.  default: 100.
% <sz> (optional) is the size of each bootstrap sample.  default: length(v).
%   if -X, use round(X*length(v)).
%
% draw bootstrap samples and then apply <fun>.
% return results in a 1 x <num> vector.
%
% we use parfor to speed things up.
%
% example:
% std(bootstrap(randn(1,1000),@mean)) < 2*1/sqrt(1000)

% NOTE: some overlap with bootstrp.m

% input
if ~exist('num','var') || isempty(num)
  num = 100;
end
if ~exist('sz','var') || isempty(sz)
  sz = length(v);
end

% calc
len = length(v);
if sz < 0
  sz = round(-sz*len);
end
v = flatten(v);

% do it
dist = zeros(1,num);
parfor p=1:num
  ix = ceil(rand(1,sz) * len);
  dist(p) = feval(fun,v(ix));
end



%%%fprintf('bootstrap');
  %%%statusdots(p,num);
%%%fprintf('done.\n');
