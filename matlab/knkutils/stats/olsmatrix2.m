function f = olsmatrix2(X)

% function f = olsmatrix2(X)
%
% <X> is samples x parameters
%
% what we want to do is to perform OLS regression using <X>
% and obtain the parameter estimates.  this is accomplished
% by inv(X'*X)*X'*y = f*y where y is the data (samples x cases).
%
% what this function does is to return <f> which has dimensions
% parameters x samples.
%
% we check for a special case, namely, when one or more regressors 
% are all zeros.  if we find that this is the case, we issue a warning
% and simply ignore these regressors when fitting.  thus, the weights
% associated with these regressors will be zeros.
%
% if any warning messages are produced by the inversion process, then we die.
% this is a conservative strategy that ensures that the regression is 
% well-behaved (i.e. has a unique, finite solution).  (note that this does
% not cover the case of zero regressors, which is gracefully handled as
% described above.)
%
% note that no scale normalization of the regressor columns is performed.
% also, note that we use \ to perform the inversion.
%
% see also olsmatrix.m.
%
% history:
% 2013/05/12 - change how zero-regressors are handled: if zero regressors are
%              found, give a warning and ensure that zero weights are assigned to
%              them (thus, we no longer result in a crash).

% check
assert(all(~isnan(X(:))));

% bad regressors are those that are all zeros
bad = all(X==0,1);
good = ~bad;

% report warning
if any(bad)
  warning('One or more regressors are all zeros; we will estimate a 0 weight for those regressors.');
end

% do it
if any(bad)

  f = zeros(size(X,2),size(X,1));
  lastwarn('');
  f(good,:) = (X(:,good)'*X(:,good))\X(:,good)';
  assert(isempty(lastwarn),lastwarn);

else

  lastwarn('');
  f = (X'*X)\X';
  assert(isempty(lastwarn),lastwarn);

end
