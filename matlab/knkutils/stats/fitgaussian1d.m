function [params,R2] = fitgaussian1d(x,y,params0,fixparams)

% function [params,R2] = fitgaussian1d(x,y,params0,fixparams)
%
% <x>,<y> are row vectors of the same size.  <x> specifies x-coordinates
%   and <y> specifies values.  if <x> is [], default to 1:length(<y>).
% <params0> (optional) is an initial seed.
%   default is [] which means make up our own initial seed.
%   TODO: our initial seed is fairly dumb; we should implement a smarter one!
% <fixparams> (optional) is
%   0 means do nothing special
%   1 means fix the offset to 0
%   default: 0.
%
% use lsqcurvefit.m to estimate parameters of a 1D Gaussian function.
% return:
%  <params> is like the input to evalgaussian1d.m
%  <R2> is the R^2 between fitted and actual y-values (see calccod.m).
%
% example:
% xx = 1:.1:10;
% yy = evalgaussian1d([5 1 4 0],xx);
% yy = yy + 0.1*randn(1,length(yy));
% [params,R2] = fitgaussian1d(xx,yy);
% figure; hold on;
% plot(xx,yy,'ro-');
% plot(xx,evalgaussian1d(params,xx),'b-');

% input
if ~exist('params0','var') || isempty(params0)
  params0 = [];
end
if ~exist('fixparams','var') || isempty(fixparams)
  fixparams = 0;
end

% construct coordinates
if isempty(x)
  x = 1:length(y);
end

% define options
options = optimset('Display','iter','FunValCheck','on','MaxFunEvals',Inf,'MaxIter',Inf,'TolFun',1e-6,'TolX',1e-6);

% define seed
if isempty(params0)
  params0 = [(min(x)+max(x))/2 (max(x)-min(x))/2 0 (min(y)+max(y))/2];
end

% deal with fixing
switch fixparams
case 0
  ix = 1:4;
case 1
  ix = [1:3];
  params0(4) = 0;
end

% report
fprintf('initial seed is %s\n',mat2str(params0,5));

% define bounds
%              m    s    g    d
paramslb = [-Inf    0 -Inf -Inf];
paramsub = [ Inf  Inf  Inf  Inf];

% do it
[params,d,d,exitflag,output] = lsqcurvefit(@(pp,xx) ...
  nanreplace(evalgaussian1d(copymatrix(params0,ix,pp),xx),0,2),params0(ix),x,y,paramslb(ix),paramsub(ix),options);
assert(exitflag > 0);
params = copymatrix(params0,ix,params);

% how well did we do?
R2 = calccod(evalgaussian1d(params,x),y);
