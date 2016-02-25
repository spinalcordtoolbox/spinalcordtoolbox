function [params,R2] = fitgaussian3d(m,params0,fixparams)

% function [params,R2] = fitgaussian3d(m,params0,fixparams)
%
% <m> is a 3D matrix
% <params0> (optional) is an initial seed.
%   default is [] which means make up our own initial seed.
% <fixparams> (optional) is
%   0 means do nothing special
%   1 means fix the offset to 0
%   default: 0.
%
% use lsqcurvefit.m to estimate parameters of a 3D Gaussian function.
% return:
%  <params> is like the input to evalgaussian3d.m
%  <R2> is the R^2 between fitted and actual m-values (see calccod.m).
%
% example:
% vol = processmulti(@(x) imresize(x,[64 64]),getsamplebrain(2));
% [params,R2] = fitgaussian3d(vol);
% figure; imagesc(makeimagestack(vol,.1));
% [xx,yy,zz] = ndgrid(1:64,1:64,1:16);
% figure; imagesc(makeimagestack(evalgaussian3d(params,xx,yy,zz))); title(sprintf('R2=%.5f',R2));
%
% history:
% 2010/10/02 - new initial seed. more liberal tolerance. explicitly guard against nans.

% input
if ~exist('params0','var') || isempty(params0)
  params0 = [];
end
if ~exist('fixparams','var') || isempty(fixparams)
  fixparams = 0;
end

% construct coordinates
[xx,yy,zz] = ndgrid(1:size(m,1),1:size(m,2),1:size(m,3));

% define options
options = optimset('Display','iter','FunValCheck','on','MaxFunEvals',Inf,'MaxIter',Inf,'TolFun',1e-6,'TolX',1e-6);

% define seed
if isempty(params0)
  params0 = [(1+size(m,1))/2 (1+size(m,2))/2 (1+size(m,3))/2 size(m,1)/5 size(m,2)/5 size(m,3)/5 iqr(m(:)) mean(m(:)) 1];
end

% deal with fixing
switch fixparams
case 0
  ix = 1:9;
case 1
  ix = [1:7 9];
  params0(8) = 0;
end

% report
fprintf('initial seed is %s\n',mat2str(params0,5));

% define bounds
%             mx   my   mz  sx  sy  sz    g    d   n
paramslb = [-Inf -Inf -Inf   0   0   0 -Inf -Inf   0];
paramsub = [ Inf  Inf  Inf Inf Inf Inf  Inf  Inf Inf];

% do it
[params,d,d,exitflag,output] = lsqcurvefit(@(pp,xx)nanreplace(evalgaussian3d(copymatrix(params0,ix,pp),xx),0,2),params0(ix),[flatten(xx); flatten(yy); flatten(zz)],flatten(m),paramslb(ix),paramsub(ix),options);
assert(exitflag > 0);
params = copymatrix(params0,ix,params);

% how well did we do?
R2 = calccod(evalgaussian3d(params,[flatten(xx); flatten(yy); flatten(zz)]),flatten(m));
