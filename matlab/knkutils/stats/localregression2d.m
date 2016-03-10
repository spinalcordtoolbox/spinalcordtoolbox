function f = localregression2d(x,y,w,x0,y0,degree,kernel,h,wts,mode)

% function f = localregression2d(x,y,w,x0,y0,degree,kernel,h,wts,mode)
%
% <x>,<y>,<w> are matrices of the same size with the data
% <x0>,<y0> are matrices of the same size with the points to evaluate at
% <degree> (optional) is 0 or 1.  default: 1.
% <kernel> (optional) is 'epan'.  default: 'epan'.
% <h> (optional) is the bandwidth [xb yb].  values can be Inf.
%   can be a scalar in which case we use that for both dimensions.
%   default: [std(x(:)) std(y(:))]/10.
% <wts> (optional) is a matrix the same size as <w> with non-negative numbers.
%   these are weights that are applied to the local regression in order to
%   allow certain points to have more influence than others.  note that
%   the weights enter the regression in exactly the same way as the kernel
%   weights.  default: ones(size(<w>)).
% <mode> (optional) is
%   0 means normal
%   1 means that <x>,<y> are generated from ndgrid(1:nx,1:ny)
%     and that <w> and <wts> are matrices of the same size as <x>,<y>.
%     we require that there be no NaNs in <w> and <wts>.
%     the point of this mode is to speed up execution.
%   default: 0.
%
% return a matrix with the value of the function at <x0>,<y0>.
%
% singular warnings are suppressed.  can return NaNs.
% note that entries with NaN in <x>, <y>, or <w> are ignored.
%
% see also localregression.m, localregression3d.m, and localregression4d.m.
%
% note that we use parfor as a way to potentially speed up execution.
%
% example:
% im = randn(100,100);
% [xx,yy] = ndgrid(1:100,1:100);
% im2 = localregression2d(xx,yy,im,xx,yy,[],[],3,[],1);
% figure; imagesc(im); caxis([-3 3]);
% figure; imagesc(im2); caxis([-3 3]);

% input
if ~exist('degree','var') || isempty(degree)
  degree = 1;
end
if ~exist('kernel','var') || isempty(kernel)
  kernel = 'epan';
end
if ~exist('h','var') || isempty(h)
  h = [std(x(:)) std(y(:))]/10;
end
if ~exist('wts','var') || isempty(wts)
  wts = ones(size(w));
  wtsopt = 1;
else
  wtsopt = 0;
end
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end
if length(h)==1
  h = repmat(h,[1 2]);
end

% prep
switch mode
case 0

  % deal with NaN (might become a row vector)
  bad = isnan(x) | isnan(y) | isnan(w) | isnan(wts);
  x(bad) = [];
  y(bad) = [];
  w(bad) = [];
  wts(bad) = [];
  x = vflatten(x);
  y = vflatten(y);
  w = vflatten(w);
  wts = vflatten(wts);
  
  % need to avoid crashing with parfor (weird bug)
  nx = [];
  ny = [];

case 1

  % calc
  nx = size(x,1);
  ny = size(x,2);
  
end

  prev = warning('query'); warning('off');

% do it
f = NaN*zeros(size(x0));
parfor pp=1:numel(x0)

  % calculate k and ix
  switch mode
  case 0

    % figure out limited support and calculate kernel weights
    ixx = findlocal([x0(pp) y0(pp)],[x'; y'],h);
    temp = (abs(x(ixx)-x0(pp))/h(1)).^2 + (abs(y(ixx)-y0(pp))/h(2)).^2;
    good = find(temp <= 1);
    k = vflatten(0.75*(1-temp(good)));  % o x 1
    ix = vflatten(ixx(good));  % o x 1

    % filter out
    xA = x(ix);
    yA = y(ix);
    wA = w(ix);
    if wtsopt
      wtsA = ones(length(xA),1);
    else
      wtsA = wts(ix);
    end
    n = length(xA);
  
  case 1

    % figure out where the subvolume is
    indices = {max(1,ceil(x0(pp)-h(1))):min(nx,floor(x0(pp)+h(1))) ...
               max(1,ceil(y0(pp)-h(2))):min(ny,floor(y0(pp)+h(2)))};
    ix = false(nx,ny);
    ix(indices{:}) = true;  % this is a logical matrix that will return the subvolume elements
  
    % calculate kernel weights
    temp = bsxfun(@plus,reshape(((indices{1} - x0(pp))/h(1)).^2,[],1), ...
                        reshape(((indices{2} - y0(pp))/h(2)).^2,1,[]));
    k = 0.75*(1-temp);
    k(k<0) = 0;
    k = vflatten(k);  % o x 1 (length is number of elements in subvolume)

    % filter out
    numx = indices{1}(end)-indices{1}(1)+1;
    numy = indices{2}(end)-indices{2}(1)+1;
    xA = vflatten(repmat(reshape(indices{1},[numx 1]),[1 numy]));
    yA = vflatten(repmat(reshape(indices{2},[1 numy]),[numx 1]));
    wA = w(ix);
    if wtsopt
      wtsA = ones(length(xA),1);
    else
      wtsA = wts(ix);
    end
    n = length(xA);

  end
  
  % form X matrices
  if degree==0
    X = ones(n,1);
    x0X = [1];
  else
    X = [xA yA ones(n,1)];
    x0X = [x0(pp) y0(pp) 1];
  end

% AVOID THIS FOR SPEED REASONS
%  % form W matrix
%  W = diag(k);
  
  % solve it
  k = k .* wtsA;
    warning('off');  % hopefully this doesn't slow things down.
  sol = (X'*(repmat(k,[1 size(X,2)]).*X)) \ (X'*(k.*wA));  % LIKE THIS FOR SPEED
  if isfinite(sol)
    f(pp) = x0X*sol;
  end

end

  warning(prev);
