function [params,gains,rs,basis] = fit3dpolynomialmodel(data,mask,degree,basis,tol)

% function [params,gains,rs,basis] = fit3dpolynomialmodel(data,mask,degree,basis,tol)
%
% <data> is a 3D matrix (there must be at least 2 elements along each dimension)
% <mask> is a 3D matrix the same size as <data> but with
%   elements that are non-negative integers 0,1,2,...,P
%   where P >= 1.  the elements should completely cover
%   either the range [0,P] or [1,P].
% <degree> is the maximum polynomial degree desired
% <basis> (optional) is a speed-up
% <tol> (optional) is the tolerance to use for TolFun and TolX.  default: 1e-10.
%   can be [A B] where A is the tolerance on TolFun and B is the tolerance on TolX.
%
% use polynomial basis functions to fit <data>.  we try to
% fit the elements marked with a 1 in <mask>, then 2 in <mask>,
% and so forth.  we enforce the constraint that different cases
% have the same surface shape but allow each case to have its 
% own gain factor.  see constructpolynomialmatrix3d.m for details 
% on the basis functions.
%
% note that when max(mask(:)) is 1, we can use OLS linear regression;
% otherwise, we are forced to use lsqnonlin.m.
%
% return:
%  <params> is 1 x n with the linear weights on the various basis functions.
%  <gains> is 1 x p with the gain factors for the different cases.
%    the first gain factor is always fixed to 1 (otherwise, there is an
%    ambiguity with respect to the gains intrinsic to <params>).
%  <rs> is 1 x p with Pearson's correlation between the data and the model fit
%    for each of the p cases.
%  <basis> is a speed-up.  you can reuse <basis> as long as the size of <data>
%    is the same and as long as <mask> and <degree> are the same.  <basis> is 
%    a cell vector of things, each of which is elements x basis functions.
%    each thing in <basis> corresponds to a positive integer into <mask>.
%    the first thing in <basis> represents the 1s in <mask>, the second
%    represents the 2s in <mask>, and so on.
%
% see also fit2dpolynomialmodel.m and fit3dpolynomialmodel2.m.
%
% example:
% data = convn(randn(30,30,30),ones(10,10,10),'same');
% [params,gains,rs,basis] = fit3dpolynomialmodel(data,ones(30,30,30),6);
% figure; imagesc(makeimagestack(reshape(data,30,30,30)));
% figure; imagesc(makeimagestack(reshape(basis{1}*params',30,30,30)));

% input
if ~exist('basis','var') || isempty(basis)
  basis = [];
end
if ~exist('tol','var') || isempty(tol)
  tol = 1e-10;
end
if length(tol) == 1
  tol = repmat(tol,[1 2]);
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
  basis = {};
  for p=1:maxnum
    basis{p} = constructpolynomialmatrix3d(datasize,find(mask==p),degree);
  end
end
nbasis = size(basis{1},2);

% fit it
if maxnum == 1
  params = (olsmatrix(basis{1})*data0{1})';
  gains = [1];
else
  options = optimset('Display','iter','MaxFunEvals',Inf,'MaxIter',Inf,'TolFun',tol(1),'TolX',tol(2));
  [params,d,d,exitflag,output] = ...
    lsqnonlin(@(params) cat(1,data0{:}) - blkdiag(basis{:})*vflatten(params(1:nbasis)'*[1 params(end-(maxnum-1)+1:end)]), ...
              [ones(1,nbasis) ones(1,maxnum-1)],[],[],options);
  assert(exitflag >= 0);

  % prep output
  gains = [1 params(end-(maxnum-1)+1:end)];
  params = params(1:nbasis);
end

% calc goodness
rs = [];
for p=1:maxnum
  rs(p) = calccorrelation(data0{p},basis{p}*params');
end
