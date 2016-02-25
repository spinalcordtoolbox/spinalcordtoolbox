function [params,R2] = fitgaussian2d(m,params0,fixparams)

% function [params,R2] = fitgaussian2d(m,params0,fixparams)
%
% <m> is a 2D matrix
% <params0> (optional) is an initial seed.
%   default is [] which means make up our own initial seed.
% <fixparams> (optional) is
%   0 means do nothing special
%   1 means fix the offset to 0
%   2 means fix the offset to 0 and constrain the Gaussian to be isotropic
%   default: 0.
%
% use lsqcurvefit.m to estimate parameters of a 2D Gaussian function.
% return:
%  <params> is like the input to evalgaussian2d.m
%  <R2> is the R^2 between fitted and actual m-values (see calccod.m).
%
% note that the parameters are in the matrix coordinate frame (see example).
%
% example:
% [xx,yy] = meshgrid(1:30,1:20);  % 1:30 in x-direction, 1:20 in y-direction
% im = evalgaussian2d([7 12 3 3 10 0],xx,yy) + 0.5*randn(20,30);
% [params,R2] = fitgaussian2d(im,[],2);
% figure; imagesc(im);
% figure; imagesc(evalgaussian2d(params,xx,yy)); title(sprintf('R2=%.5f',R2));

% input
if ~exist('params0','var') || isempty(params0)
  params0 = [];
end
if ~exist('fixparams','var') || isempty(fixparams)
  fixparams = 0;
end

% construct coordinates
[xx,yy] = meshgrid(1:size(m,2),1:size(m,1));

% define options
options = optimset('Display','iter','FunValCheck','on','MaxFunEvals',Inf,'MaxIter',Inf,'TolFun',1e-6,'TolX',1e-6);

% define seed
if isempty(params0)
  params0 = [(1+size(m,2))/2 (1+size(m,1))/2 size(m,2)/5 size(m,1)/5 iqr(m(:)) mean(m(:))];
end

% deal with fixing.
%   ix are the indices that we are optimizing.
%   ixsp are special indices in params0 to fill in to make a full-fledged parameter.
%   ppi are indices to pull out from pp to fill in ixsp with.
switch fixparams
case 0
  ix = 1:6;
  ixsp = ix;
  ppi = 1:6;
case 1
  ix = 1:5;
  ixsp = ix;
  ppi = 1:5;
  params0(6) = 0;  % explicitly set
case 2
  ix = [1:3 5];
  ixsp = [1:5];
  ppi = [1 2 3 3 4];
  params0([4 6]) = 0;  % explicitly set
end

% report
fprintf('initial seed is %s\n',mat2str(params0,5));

% define bounds
%             mx   my  sx  sy    g    d
paramslb = [-Inf -Inf   0   0 -Inf -Inf];
paramsub = [ Inf  Inf Inf Inf  Inf  Inf];

% do it
[params,d,d,exitflag,output] = lsqcurvefit(@(pp,xx)nanreplace(evalgaussian2d(copymatrix(params0,ixsp,pp(ppi)),xx),0,2),params0(ix),[flatten(xx); flatten(yy)],flatten(m),paramslb(ix),paramsub(ix),options);
assert(exitflag > 0);
params = copymatrix(params0,ixsp,params(ppi));

% how well did we do?
R2 = calccod(evalgaussian2d(params,[flatten(xx); flatten(yy)]),flatten(m));
