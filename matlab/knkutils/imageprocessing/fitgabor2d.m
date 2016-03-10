function [params,R2] = fitgabor2d(z,params0)

% function [params,R2] = fitgabor2d(z,params0)
%
% <z> is a 2D square matrix of values
% <params0> (optional) is an initial seed.
%   default is [] which means make up our own initial seed.
%
% use lsqcurvefit.m to estimate parameters of a 2D Gabor function.
% return:
%  <params> is like the input to evalgabor2d.m (the second flavor).
%  <R2> is the R^2 between fitted and actual z-values (see calccod.m).
%
% example:
% z = makegabor2d(32,[],[],4,pi/6,0,2) + randn(32,32)*.1;
% [params,R2] = fitgabor2d(z);
% figure; imagesc(z,[-2 2]);
% [xx,yy] = calcimagecoordinates(32);
% figure; imagesc(evalgabor2d(params,xx,yy),[-2 2]); title(sprintf('R2=%.5f',R2));
%
% history:
% 2010/06/15 - switch to calccod.m

% input
if ~exist('params0','var') || isempty(params0)
  params0 = [];
end

% calc
res = size(z,1);

% construct coordinates
[xx,yy] = calcimagecoordinates(res);

% define options
  % 'OutputFcn',@fitgaussian2doutput);,  % ,,OutputFcn',@fitblah
options = optimset('Display','final','MaxFunEvals',Inf,'MaxIter',Inf,'TolFun',1e-10,'TolX',1e-10);

% define seed
if isempty(params0)
  [cxx,cyy] = calccpfov(res);
  [mx,ix] = max(flatten(fftshift(abs(fft2(z)))));
  cpfov = sqrt(cxx(ix)^2+cyy(ix)^2);
  xseed = sum(flatten(bsxfun(@times,1:res,z.^2)))/sum(z(:).^2);
  yseed = sum(vflatten(bsxfun(@times,fliplr(1:res),z.^2)))/sum(z(:).^2);
  orseed = mod(-atan2(cyy(ix),cxx(ix))-pi/2,pi);
  params0 =  [cpfov/res orseed 0 xseed yseed (2/cpfov)/4*res (2/cpfov)/4*res range(z(:))/2 mean(z(:))];
  fprintf('initial seed is %s\n',mat2str(params0,5));
end

% define bounds
%           cpu          ang  phase  mx    my   sd1   sd2    g             d
paramslb = [0           -Inf   -Inf -Inf  -Inf  -Inf  -Inf   0            -Inf];
paramsub = [1/2*sqrt(2)  Inf    Inf  Inf   Inf   Inf   Inf   Inf           Inf ];

% do it
[params,d,d,exitflag,output] = lsqcurvefit(@evalgabor2d,params0,[flatten(xx); flatten(yy)],flatten(z),paramslb,paramsub,options);
assert(exitflag > 0);

% sanity transformation
params(2:3) = mod(params(2:3),2*pi);
params(6:7) = abs(params(6:7));

% how well did we do?
R2 = calccod(evalgabor2d(params,[flatten(xx); flatten(yy)]),flatten(z));





% function stop = fitblah(params,optimValues,state,varargin)
% 
% % function stop = fitblah(params,optimValues,state,varargin)
% %
% % <params> is the current optimization parameter
% % <optimValues>,<state> are optimization state stuff
% % <varargin> are additional arguments we passed to lsqcurvefit.m
% %
% % return <stop> which is whether to stop optimization.
% 
% if mod(optimValues.iteration,1)==0
%   [xx,yy] = meshgrid(1:32,32:-1:1);
%   drawnow; imagesc(evalgabor2d(params,xx,yy));
%   fprintf('%s\n',mat2str(params,5));
%   pause;
% end
% 
% stop=0;
