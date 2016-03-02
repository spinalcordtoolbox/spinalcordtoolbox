function [params,r,basis] = fit3dpolynomialmodel2(data,mask,degree,basis)

% function [params,r,basis] = fit3dpolynomialmodel2(data,mask,degree,basis)
%
% <data> is a 3D matrix (there must be at least 2 elements along each dimension)
% <mask> is a 3D matrix the same size as <data> but with
%   elements that are non-negative integers 0,1,2,...,P
%   where P >= 1.  the elements should completely cover
%   either the range [0,P] or [1,P].
% <degree> is the maximum polynomial degree desired
% <basis> (optional) is a speed-up
%
% use polynomial basis functions to fit <data>.  we try to
% fit the elements marked with a 1 in <mask>, then 2 in <mask>,
% and so forth.  we enforce the constraint that different cases
% have exactly the same surface except that each case has its own
% DC offset.  to achieve this, what we do is to construct the usual
% polynomial basis functions, find the DC basis function, and then
% split the DC basis function into a separate regressor for each case.
% the basis function matrix looks like this:
%   [DC1 DC2 ... DCn other1 other2 other3 ...]
% where DC1, DC2, ..., DCn represent the DC basis function split for
% different cases and other1, other2, other3, ... represent the remaining
% basis functions.  for additional details, see constructpolynomialmatrix3d.m
% and the example below.
%
% return:
%  <params> is 1 x n with the linear weights on the various basis functions.
%  <r> is Pearson's correlation between the data and the model fit.
%  <basis> is a speed-up.  you can reuse <basis> as long as the size of <data>
%    is the same and as long as <mask> and <degree> are the same.
%
% see also fit2dpolynomialmodel.m and fit3dpolynomialmodel.m.
%
% example:
% data = convn(randn(30,30,30),ones(10,10,10),'same');
% [params,r] = fit3dpolynomialmodel2(data,(rand(30,30,30) > .5)+1,2);
% basis = constructpolynomialmatrix3d([30 30 30],find(ones(30,30,30)),2);
% dcix = find(all(basis==1,1));
% figure; imagesc(makeimagestack(reshape(data,30,30,30)));
% figure; imagesc(makeimagestack(reshape(basis(:,setdiff(1:size(basis,2),dcix))*params(3:end)',30,30,30)));

% input
if ~exist('basis','var') || isempty(basis)
  basis = [];
end

% calc
mask = double(mask);  % make sure not logical
datasize = sizefull(data,3);
maxnum = max(mask(:));

% prep data
data0 = {};
for p=1:maxnum
  data0{p} = data(mask==p);
end

% prep basis functions
if isempty(basis)

  % make them
  basis = {};
  for p=1:maxnum
    basis{p} = constructpolynomialmatrix3d(datasize,find(mask==p),degree);
  end
  
  % do the DC hackery
  dcix = find(all(basis{1}==1,1));  % find the DC basis function
  temp = cellfun(@(x) x(:,dcix),basis,'UniformOutput',0);  % extract the DC from each case
  temp2 = cellfun(@(x) x(:,setdiff(1:size(basis{1},2),dcix)),basis,'UniformOutput',0);  % extract the other basis functions from each case
  basis = cat(2,blkdiag(temp{:}),cat(1,temp2{:}));  % the final basis consists of separate DCs for each case and then all other basis functions

end

% fit it
params = (olsmatrix(basis)*cat(1,data0{:}))';

% calc goodness
r = calccorrelation(cat(1,data0{:}),basis*params');
