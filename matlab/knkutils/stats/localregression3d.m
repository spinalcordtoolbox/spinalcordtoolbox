function f = localregression3d(x,y,z,w,x0,y0,z0,degree,kernel,h,wts,mode)

% function f = localregression3d(x,y,z,w,x0,y0,z0,degree,kernel,h,wts,mode)
%
% <x>,<y>,<z>,<w> are matrices of the same size with the data
% <x0>,<y0>,<z0> are matrices of the same size with the points to evaluate at
% <degree> (optional) is 0 or 1.  default: 1.
% <kernel> (optional) is 'epan'.  default: 'epan'.
% <h> (optional) is the bandwidth [xb yb zb].  values can be Inf.
%   can be a scalar in which case we use that for all three dimensions.
%   default: [std(x(:)) std(y(:)) std(z(:))]/10.
% <wts> (optional) is a matrix the same size as <w> with non-negative numbers.
%   these are weights that are applied to the local regression in order to
%   allow certain points to have more influence than others.  note that
%   the weights enter the regression in exactly the same way as the kernel
%   weights.  default: ones(size(<w>)).
% <mode> (optional) is
%   0 means normal
%   1 means that <x>,<y>,<z> are generated from ndgrid(1:nx,1:ny,1:nz)
%     and that <w> and <wts> are matrices of the same size as <x>,<y>,<z>.
%     we require that there be no NaNs in <w> and <wts>.
%     the point of this mode is to speed up execution.
%   default: 0.
%
% return a matrix with the value of the function at <x0>,<y0>,<z0>.
%
% singular warnings are suppressed.  can return NaNs.
% note that entries with NaN in <x>, <y>, <z>, or <w> are ignored.
%
% see also localregression.m and localregression4d.m.
%
% note that we use parfor as a way to potentially speed up execution.
% if parallelization is used, note that status dots are outputted only at the end.
%
% example:
% x = randn(1,1000);
% y = randn(1,1000);
% z = randn(1,1000);
% w = sin(x) + cos(y) + tan(z) + .2*randn(size(x));
% [x0,y0,z0] = ndgrid(-1:.1:1);
% w0 = localregression3d(x,y,z,w,flatten(x0),flatten(y0),flatten(z0));
% w0actual = flatten(sin(x0) + cos(y0) + tan(z0));
% figure;
% scatter(w0,w0actual,'r.');
% axissquarify;
% xlabel('local regression fit'); ylabel('true values');

% input
if ~exist('degree','var') || isempty(degree)
  degree = 1;
end
if ~exist('kernel','var') || isempty(kernel)
  kernel = 'epan';
end
if ~exist('h','var') || isempty(h)
  h = [std(x(:)) std(y(:)) std(z(:))]/10;
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
  h = repmat(h,[1 3]);
end

% prep
switch mode
case 0

  % deal with NaN (might become a row vector)
  bad = isnan(x) | isnan(y) | isnan(z) | isnan(w) | isnan(wts);
  x(bad) = [];
  y(bad) = [];
  z(bad) = [];
  w(bad) = [];
  wts(bad) = [];
  x = vflatten(x);
  y = vflatten(y);
  z = vflatten(z);
  w = vflatten(w);
  wts = vflatten(wts);
  
  % this seems to be necessary only because of the parfor. weird.
  nx = []; ny = []; nz = [];

case 1

  % calc
  nx = size(x,1);
  ny = size(x,2);
  nz = size(x,3);
  
end

  prev = warning('query'); warning('off');

% do it
f = NaN*zeros(size(x0));
fprintf('localregression3d');
parfor pp=1:numel(x0)
  statusdots(pp,numel(x0),20);

  % calculate k and ix
  switch mode
  case 0

    % figure out limited support and calculate kernel weights
    ixx = findlocal([x0(pp) y0(pp) z0(pp)],[x'; y'; z'],h);
    temp = (abs(x(ixx)-x0(pp))/h(1)).^2 + (abs(y(ixx)-y0(pp))/h(2)).^2 + (abs(z(ixx)-z0(pp))/h(3)).^2;
    good = find(temp <= 1);
    k = vflatten(0.75*(1-temp(good)));  % o x 1
    ix = vflatten(ixx(good));  % o x 1

% OLD WAY (now we use findlocal)
%     % figure out limited support and calculate kernel weights [LIKE THIS FOR SPEED]
%     xa = abs(x-x0(pp));       % TAKES MOST TIME! WE SHOULD MAKE THIS FASTER
%     good1 = find(xa <= h(1)); % TAKES MOST TIME! WE SHOULD MAKE THIS FASTER
%     ya = abs(y(good1)-y0(pp));
%     good2 = find(ya <= h(2));
%     za = abs(z(good1(good2))-z0(pp));
%     good3 = find(za <= h(3));
%     temp = (xa(good1(good2(good3)))/h(1)).^2 + (ya(good2(good3))/h(2)).^2 + (za(good3)/h(3)).^2;
%   %%%  temp = ((x-x0(pp))/h(1)).^2 + ((y-y0(pp))/h(2)).^2 + ((z-z0(pp))/h(3)).^2;  % 1 x n   [LIKE THIS FOR SPEED]
%     good4 = find(temp <= 1);
%     k = vflatten(0.75*(1-temp(good4)));  % o x 1
%     ix = vflatten(good1(good2(good3(good4))));  % o x 1

    % get out early
    if isempty(ix)
      continue;
    end
  
    % filter out
    xA = x(ix);
    yA = y(ix);
    zA = z(ix);
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
               max(1,ceil(y0(pp)-h(2))):min(ny,floor(y0(pp)+h(2))) ...
               max(1,ceil(z0(pp)-h(3))):min(nz,floor(z0(pp)+h(3)))};
    ix = false(nx,ny,nz);
    ix(indices{:}) = true;  % this is a logical matrix that will return the subvolume elements
  
    % calculate kernel weights
    temp = bsxfun(@plus,reshape(((indices{1} - x0(pp))/h(1)).^2,[],1), ...
                        reshape(((indices{2} - y0(pp))/h(2)).^2,1,[]));
    temp = bsxfun(@plus,temp, ...
                        reshape(((indices{3} - z0(pp))/h(3)).^2,1,1,[]));
    k = 0.75*(1-temp);
    k(k<0) = 0;
    k = vflatten(k);  % o x 1 (length is number of elements in subvolume)

    % get out early
    if isempty(k)
      continue;
    end

    % filter out
    numx = indices{1}(end)-indices{1}(1)+1;
    numy = indices{2}(end)-indices{2}(1)+1;
    numz = indices{3}(end)-indices{3}(1)+1;
    xA = vflatten(repmat(reshape(indices{1},[numx 1]),[1 numy numz]));
    yA = vflatten(repmat(reshape(indices{2},[1 numy]),[numx 1 numz]));
    zA = vflatten(repmat(reshape(indices{3},[1 1 numz]),[numx numy 1]));
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
    X = [xA yA zA ones(n,1)];
    x0X = [x0(pp) y0(pp) z0(pp) 1];
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
fprintf('done.\n');

  warning(prev);
