function [params,R2] = fitrbf2d(x,y,z,params0)

% function [params,R2] = fitrbf2d(x,y,z,params0)
%
% <x>,<y>,<z> are matrices of values
% <params0> (optional) is an initial seed.
%   default is [] which means make up our own initial seed.
%
% use lsqcurvefit.m to estimate parameters of a 2D RBF.
% return:
%  <params> is like the input to evalrbf2d.m.
%  <R2> is the R^2 between fitted and actual z-values (see calccod.m).
%
% example:
% x = randn(1,1000);
% y = randn(1,1000);
% z = exp(-((x-.1).^2+(y-.3).^2)) + .2*randn(1,1000);
% [params,R2] = fitrbf2d(x,y,z);
% figure; scatter(x,y,40,z,'filled'); axis equal tight; axis([-2 2 -2 2]);
% [xx,yy] = meshgrid(-2:.1:2,-2:.1:2);
% figure; contourf(xx,yy,evalrbf2d(params,xx,yy)); axis equal tight; axis([-2 2 -2 2]); title(sprintf('R2=%.5f',R2)); 
%
% history:
% 2010/06/15 - switch to calccod.m.

% input
if ~exist('params0','var') || isempty(params0)
  params0 = [];
end

% define options
  % 'OutputFcn',@fitgaussian2doutput);,  % ,,OutputFcn',@fitblah
options = optimset('Display','final','MaxFunEvals',Inf,'MaxIter',Inf,'TolFun',1e-10,'TolX',1e-10);

% define seed
if isempty(params0)
  [mx,ix] = max(abs(z(:)));
  params0 = [x(ix) y(ix) 1 range(z(:)) min(z(:))];
  fprintf('initial seed is %s\n',mat2str(params0,5));
end

% define bounds
%           cx   cy   b   g    d
paramslb = [-Inf -Inf 0   -Inf -Inf];
paramsub = [Inf  Inf  Inf Inf  Inf];

% do it
[params,d,d,exitflag,output] = lsqcurvefit(@evalrbf2d,params0,[flatten(x); flatten(y)],flatten(z),paramslb,paramsub,options);
assert(exitflag > 0);

% how well did we do?
R2 = calccod(evalrbf2d(params,[flatten(x); flatten(y)]),flatten(z));
