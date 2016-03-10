function [params,R2] = fitorientedgaussian2d(z,ors,params0)

% function [params,R2] = fitorientedgaussian2d(z,ors,params0)
%
% <z> is a 2D square matrix of values, with different orientations along the third dimension
% <ors> is a vector of orientations corresponding to the third dimension of <z>
% <params0> (optional) is an initial seed.
%   default is [] which means make up our own initial seed.
%   note that there is some randomness in the seed.
%
% use lsqcurvefit.m to estimate parameters of an oriented 2D Gaussian function.
% return:
%  <params> is like the input to evalorientedgaussian2d.m.
%  <R2> is the R^2 between fitted and actual z-values (see calccod.m).
%
% example:
% ors = linspacecircular(0,pi,8);
% z = makeorientedgaussian2d(32,[],[],pi/6,6,3,2,ors);
% z = z + randn(size(z))*.1;
% [params,R2] = fitorientedgaussian2d(z,ors);
% figure; imagesc(makeimagestack(z,[],[],-1),[0 1]); axis equal tight;
% [xx,yy] = calcimagecoordinates(32);
% figure; imagesc(makeimagestack(evalorientedgaussian2d(params,xx,yy,ors),[],[],-1),[0 1]); axis equal tight; title(sprintf('R2=%.5f',R2));
%
% history:
% 2010/06/15 - switch to calccod.m.

% input
if ~exist('params0','var') || isempty(params0)
  params0 = [];
end

% calc
res = size(z,1);

% construct coordinates
[xx,yy] = calcimagecoordinates(res);

% define options
options = optimset('Display','final','MaxFunEvals',Inf,'MaxIter',Inf,'TolFun',1e-10,'TolX',1e-10);

% define seed
if isempty(params0)
  [mx,ix] = max(flatten(max(max(z,[],1),[],2)));
  xseed = sum(flatten(bsxfun(@times,1:res,z.^2)))/sum(z(:).^2);
  yseed = sum(vflatten(bsxfun(@times,fliplr(1:res),z.^2)))/sum(z(:).^2);
  params0 = [xseed yseed ors(ix) res/(2+rand) res/(2+rand) rand+2 range(z(:)) mean(z(:))];
  fprintf('initial seed is %s\n',mat2str(params0,5));
end

% define bounds
%           mx   my   ang  sd1  sd2  orsd g   d
paramslb = [-Inf -Inf -Inf -Inf -Inf 0    0   -Inf];
paramsub = [Inf  Inf  Inf  Inf  Inf  Inf  Inf Inf ];

% do it
[params,d,d,exitflag,output] = lsqcurvefit(@(a,b) flatten(evalorientedgaussian2d(a,b,[],ors)),params0,[flatten(xx); flatten(yy)],flatten(z),paramslb,paramsub,options);
assert(exitflag > 0);

% sanity transformation
params(3) = mod(params(3),2*pi);
params(4:5) = abs(params(4:5));

% how well did we do?
R2 = calccod(flatten(evalorientedgaussian2d(params,[flatten(xx); flatten(yy)],[],ors)),flatten(z));
