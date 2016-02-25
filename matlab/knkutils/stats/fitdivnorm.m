function [params,R2] = fitdivnorm(x,y,params0,fixparams,suppress)

% function [params,R2] = fitdivnorm(x,y,params0,fixparams,suppress)
%
% <x>,<y> are row vectors of the same size.  <x> specifies x-coordinates
%   and <y> specifies values.  if <x> is [], default to 1:length(<y>).
% <params0> (optional) is an initial seed.
%   default is [] which means make up our own initial seed.
%   TODO: our initial seed is fairly dumb; we should implement a smarter one!
% <fixparams> (optional) is
%   0 means do nothing special
%   1 means fix the offset to 0
%   2 means fix the offset to 0 and the exponent to 1
%   3 means fix the offset to 0 and the sigma to Inf
%   default: 0.
% <suppress> (optional) is whether to suppress output to command window.  default: 0.
%
% use lsqcurvefit.m to estimate parameters of a
% divisive-normalization function.  note that we restrict
% the n and s parameters to be positive.
%
% return:
%  <params> is like the input to evaldivnorm.m
%  <R2> is the R^2 between fitted and actual y-values (see calccod.m).
%
% example:
% xx = 0:.01:1;
% yy = evaldivnorm([2 .1 2 0],xx);
% yy = yy + 0.1*randn(1,length(yy));
% [params,R2] = fitdivnorm(xx,yy);
% figure; hold on;
% plot(xx,yy,'ro-');
% plot(xx,evaldivnorm(params,xx),'b-');

% input
if ~exist('params0','var') || isempty(params0)
  params0 = [];
end
if ~exist('fixparams','var') || isempty(fixparams)
  fixparams = 0;
end
if ~exist('suppress','var') || isempty(suppress)
  suppress = 0;
end

% construct coordinates
if isempty(x)
  x = 1:length(y);
end

% define options
options = optimset('Display',choose(suppress,'off','iter'),'FunValCheck','on','MaxFunEvals',Inf,'MaxIter',10000,'TolFun',1e-6,'TolX',1e-6);

% define seed
if isempty(params0)
  params0 = [1 (min(x)+max(x))/2 max(y)-min(y) min(y)];
end

% deal with fixing
switch fixparams
case 0
  ix = 1:4;
case 1
  ix = [1:3];
  params0(4) = 0;
case 2
  ix = [2:3]
  params0(1) = 1;
  params0(4) = 0;
case 3
  ix = [1 3];
  params0(2) = Inf;
  params0(4) = 0;
end

% report
if ~suppress
  fprintf('initial seed is %s\n',mat2str(params0,5));
end

% define bounds
%              n    s    g    d
paramslb = [   0    0 -Inf -Inf];
paramsub = [ Inf  Inf  Inf  Inf];

% do it
[params,d,d,exitflag,output] = lsqcurvefit(@(pp,xx) ...
  nanreplace(evaldivnorm(copymatrix(params0,ix,pp),xx),0,2),params0(ix),x,y,paramslb(ix),paramsub(ix),options);
assert(exitflag >= 0);
params = copymatrix(params0,ix,params);

% how well did we do?
R2 = calccod(evaldivnorm(params,x),y);
