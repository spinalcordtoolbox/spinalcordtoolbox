function [im,mov] = imagesearch(n,fun,im0,dispfun,tol)

% function [im,mov] = imagesearch(n,fun,im0,dispfun,tol)
%
% <n> is number of pixels in the image (assumed to be square with values in [0,1])
% <fun> is a function that accepts an image (1 x <n>) and returns a value
% <im0> (optional) is an initial seed.
%   default is [] which means use a random sample of Gaussian white noise.
% <dispfun> (optional) is a function that accepts an image and
%   displays it in some form.  if supplied, we make figure window 100, and then
%   <dispfun> is called at each iteration in the search (using 'OutputFcn').
%   we automatically do "axis equal tight" and "drawnow".  we also show progression
%   of the parameters in figure window 101.  in addition, we check 
%   if a new key has been pressed in either figure window and we stop if so.
%   special case is 0, which means @(x) imagesc(reshape(x,[sqrt(n) sqrt(n)]),[0 1]).
% <tol> (optional) is the tolerance to use.  default: 0.01.
%
% use fmincon.m to find an image that minimizes the output of <fun>.
% return <im> with the optimal image (1 x <n>).
% return <mov> with the progression (iters x <n>).
%
% example:
% im = flatten(imresize(getsampleimage,[32 32],'lanczos3'));
% im2 = imagesearch(32*32,@(x) vectorlength(x-im),[],0);

% input
if ~exist('im0','var') || isempty(im0)
  im0 = max(min(randn(1,n)/6 + .5,1),0);
end
if ~exist('dispfun','var') || isempty(dispfun)
  dispfun = [];
end
if ~exist('tol','var') || isempty(tol)
  tol = 0.01;
end
if isequal(dispfun,0)
  dispfun = @(x) imagesc(reshape(x,[sqrt(n) sqrt(n)]),[0 1]);
end

% prep record
global IMAGESEARCH_X
IMAGESEARCH_X = [];

% define options
alg = 'interior-point';
%alg = 'active-set'; %SLOW
%alg = 'trust-region-reflective'; DOESN"T WORK FOR THIS PROBLEM
if isempty(dispfun)
  options = optimset('Display','iter','MaxFunEvals',Inf,'MaxIter',Inf,'TolFun',tol,'TolX',tol,'Algorithm',alg, ...
                     'OutputFcn',@outfun2);
else
  global IMAGESEARCH_100K IMAGESEARCH_101K;
  figure(100); IMAGESEARCH_100K = get(100,'CurrentCharacter');
  figure(101); IMAGESEARCH_101K = get(101,'CurrentCharacter');
  options = optimset('Display','iter','MaxFunEvals',Inf,'MaxIter',Inf,'TolFun',tol,'TolX',tol,'Algorithm',alg, ...
                     'OutputFcn',@(x,y,z) outfun(x,y,z,dispfun));
end
      
% define bounds
imlb = zeros(1,n);
imub = ones(1,n);

% do it
[im,fval,exitflag,output] = fmincon(fun,im0,[],[],[],[],imlb,imub,[],options);
assert(exitflag > 0 || exitflag==-1);  % -1 means outputfcn terminated it

% output
mov = IMAGESEARCH_X;

%%%%%

function stop = outfun(x,optimValues,state,dispfun)

global IMAGESEARCH_X IMAGESEARCH_100K IMAGESEARCH_101K;

% display progress
figure(100);
feval(dispfun,x);
axis equal tight;

% show parameters
IMAGESEARCH_X = cat(1,IMAGESEARCH_X,x);
figure(101);
plot(IMAGESEARCH_X);
drawnow;

% stop iff the user pressed a key
stop = ~isequal(get(100,'CurrentCharacter'),IMAGESEARCH_100K) | ~isequal(get(101,'CurrentCharacter'),IMAGESEARCH_101K);

%%%%%

function stop = outfun2(x,optimValues,state)

global IMAGESEARCH_X;
IMAGESEARCH_X = cat(1,IMAGESEARCH_X,x);
stop = 0;
