function f = olsmatrix(X,mode)

% function f = olsmatrix(X,mode)
%
% <X> is samples x parameters
% <mode> (optional) is
%   0 means normal operation
%   1 means use inv instead of \ and omit unit-length normalization.
%     the point of this mode is to reduce memory usage (i think).
%   default: 0.
%
% what we want to do is to perform OLS regression using <X>
% and obtain the parameter estimates.  this is accomplished
% by inv(X'*X)*X'*y = f*y where y is the data (samples x cases).
%
% what this function does is to return <f> which has dimensions
% parameters x samples. 
%
% to ensure well-conditioning, we unit-length normalize each column
% of <X> before doing the inv.m operation.  also, we actually use
% left-slash (\), which apparently is equivalent to inv.m but faster
% and more accurate (see code for details).  if you pass <mode> as 1,
% we omit the normalization and use the inv method instead of the \ method.
%
% also, in the case that one or more regressors in <X> are all zero, then
% without special handling, then this case will result in warnings and
% NaN results.  what we do is to explicitly ensure that all-zero regressors
% are ignored and that there are all-zero rows for these regressors 
% in <f>.  this makes it such that the weights estimated for these 
% regressors are simply zero.
%
% see also olsmatrix2.m.
%
% history:
% 2011/06/27 - explicitly ignore 0
% 2011/03/29 - mode 1 now omits unit length normalization.
% 2011/03/28 - add <mode> input.
% 2010/08/05: now, we do not try to detect weird cases with low variance.
%             instead, we just blindly unit-length normalize each column.

% input
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end

% check
assert(all(~isnan(X(:))));

% deal with degenerate regressors
good = ~all(X==0,1);

% initialize result
f = zeros(size(X,2),size(X,1));

% do it
switch mode
case 0
  [X,len] = unitlength(X(:,good),1,[],0);
  temp = diag(1./len)*((X'*X)\X');  % hm, suggested by M-Lint
case 1
  temp = inv(X(:,good)'*X(:,good))*X(:,good)';
end

% return
if any(good)
  f(good,:) = temp;
end
