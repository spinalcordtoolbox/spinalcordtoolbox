function f = localregression4d(x,y,z,t,w,x0,y0,z0,t0,degree,kernel,h,wts,mode)

% function f = localregression4d(x,y,z,t,w,x0,y0,z0,t0,degree,kernel,h,wts,mode)
%
% <x>,<y>,<z>,<t>,<w> are matrices of the same size with the data
% <x0>,<y0>,<z0>,<t0> are matrices of the same size with the points to evaluate at
% <degree> (optional) is 0 or 1.  default: 1.
% <kernel> (optional) is 'epan'.  default: 'epan'.
% <h> (optional) is the bandwidth [xb yb zb tb].  values can be Inf.
%   can be a scalar in which case we use that for all four dimensions.
%   default: [std(x(:)) std(y(:)) std(z(:)) std(t(:))]/10.
% <wts> (optional) is a matrix the same size as <w> with non-negative numbers.
%   these are weights that are applied to the local regression in order to
%   allow certain points to have more influence than others.  note that
%   the weights enter the regression in exactly the same way as the kernel
%   weights.  default: ones(size(<w>)).
% <mode> (optional) is
%   0 means normal
%   1 means that <x>,<y>,<z>,<t> are generated from ndgrid(1:nx,1:ny,1:nz,1:nt)
%     and that <w> and <wts> are matrices of the same size as <x>,<y>,<z>,<t>.
%     we require that there be no NaNs in <w> and <wts>.
%     the point of this mode is to speed up execution.
%   default: 0.
%
% return a matrix with the value of the function at <x0>,<y0>,<z0>,<t0>.
%
% singular warnings are suppressed.  can return NaNs.
% note that entries with NaN in <x>, <y>, <z>, <t>, or <w> are ignored.
%
% see also localregression.m and localregression3d.m.
%
% note that we use parfor as a way to potentially speed up execution.
% if parallelization is used, note that status dots are outputted only at the end.
%
% example:
% x = randn(1,1000);
% y = randn(1,1000);
% z = randn(1,1000);
% t = randn(1,1000);
% w = sin(x) + cos(y) + tan(z) + t + .2*randn(size(x));
% [x0,y0,z0,t0] = ndgrid(-1:.1:1);
% w0 = localregression4d(x,y,z,t,w,flatten(x0),flatten(y0),flatten(z0),flatten(t0));
% w0actual = flatten(sin(x0) + cos(y0) + tan(z0) + t0);
% figure;
% scatter(w0,w0actual,'r.');
% axissquarify;
% xlabel('local regression fit'); ylabel('true values');
%
% another example:
% [x,y,z,t] = ndgrid(1:10,1:10,1:10,1:10);
% w = x + y + z + t;
% [x0,y0,z0,t0] = ndgrid(4:6);
% w0 = localregression4d(x,y,z,t,w,flatten(x0),flatten(y0),flatten(z0),flatten(t0),[],[],1.5,[],1);
% w0actual = flatten(x0 + y0 + z0 + t0);
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
  h = [std(x(:)) std(y(:)) std(z(:)) std(t(:))]/10;
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
  h = repmat(h,[1 4]);
end

% prep
switch mode
case 0

  % deal with NaN (might become a row vector)
  bad = isnan(x) | isnan(y) | isnan(z) | isnan(t) | isnan(w) | isnan(wts);
  x(bad) = [];
  y(bad) = [];
  z(bad) = [];
  t(bad) = [];
  w(bad) = [];
  wts(bad) = [];
  x = vflatten(x);
  y = vflatten(y);
  z = vflatten(z);
  t = vflatten(t);
  w = vflatten(w);
  wts = vflatten(wts);

case 1

  % calc
  nx = size(x,1);
  ny = size(x,2);
  nz = size(x,3);
  nt = size(x,4);
  
end
 
  prev = warning('query'); warning('off');

% do it
f = NaN*zeros(size(x0));
fprintf('localregression4d');
parfor pp=1:numel(x0)
  statusdots(pp,numel(x0),20);

  % calculate k and ix
  switch mode
  case 0
  
    % figure out limited support and calculate kernel weights
    ixx = findlocal([x0(pp) y0(pp) z0(pp) t0(pp)],[x'; y'; z'; t'],h);
    temp = (abs(x(ixx)-x0(pp))/h(1)).^2 + (abs(y(ixx)-y0(pp))/h(2)).^2 + (abs(z(ixx)-z0(pp))/h(3)).^2 + (abs(t(ixx)-t0(pp))/h(4)).^2;
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
%     ta = abs(t(good1(good2(good3)))-t0(pp));
%     good4 = find(ta <= h(4));
%     temp = (xa(good1(good2(good3(good4))))/h(1)).^2 + (ya(good2(good3(good4)))/h(2)).^2 + (za(good3(good4))/h(3)).^2 + (ta(good4)/h(4)).^2;
%   %%%  temp = ((x-x0(pp))/h(1)).^2 + ((y-y0(pp))/h(2)).^2 + ((z-z0(pp))/h(3)).^2;  % 1 x n   [LIKE THIS FOR SPEED]
%     good5 = find(temp <= 1);
%     k = vflatten(0.75*(1-temp(good5)));  % o x 1
%     ix = vflatten(good1(good2(good3(good4(good5)))));  % o x 1

    % filter out
    xA = x(ix);
    yA = y(ix);
    zA = z(ix);
    tA = t(ix);
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
               max(1,ceil(z0(pp)-h(3))):min(nz,floor(z0(pp)+h(3))) ...
               max(1,ceil(t0(pp)-h(4))):min(nt,floor(t0(pp)+h(4)))};
    ix = false(nx,ny,nz,nt);
    ix(indices{:}) = true;  % this is a logical matrix that will return the subvolume elements
  
    % calculate kernel weights
    temp = bsxfun(@plus,reshape(((indices{1} - x0(pp))/h(1)).^2,[],1), ...
                        reshape(((indices{2} - y0(pp))/h(2)).^2,1,[]));
    temp = bsxfun(@plus,temp, ...
                        reshape(((indices{3} - z0(pp))/h(3)).^2,1,1,[]));
    temp = bsxfun(@plus,temp, ...
                        reshape(((indices{4} - t0(pp))/h(4)).^2,1,1,1,[]));  % squared distance away from desired point (in the subvolume)
    k = 0.75*(1-temp);
    k(k<0) = 0;
    k = vflatten(k);  % o x 1 (length is number of elements in subvolume)

    % filter out
    numx = indices{1}(end)-indices{1}(1)+1;
    numy = indices{2}(end)-indices{2}(1)+1;
    numz = indices{3}(end)-indices{3}(1)+1;
    numt = indices{4}(end)-indices{4}(1)+1;
    xA = vflatten(repmat(reshape(indices{1},[numx 1]),[1 numy numz numt]));
    yA = vflatten(repmat(reshape(indices{2},[1 numy]),[numx 1 numz numt]));
    zA = vflatten(repmat(reshape(indices{3},[1 1 numz]),[numx numy 1 numt]));
    tA = vflatten(repmat(reshape(indices{4},[1 1 1 numt]),[numx numy numz 1]));
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
    X = [xA yA zA tA ones(n,1)];
    x0X = [x0(pp) y0(pp) z0(pp) t0(pp) 1];
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
