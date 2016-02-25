function f = randnmulti(n,mn,sigma,sd)

% function f = randnmulti(n,mn,sigma,sd)
% 
% <n> is the number of random points to draw
% <mn> is a vector with the desired mean for each variable.
%   for example, [1 2] means the mean of the Gaussian is at (1,2).
%   can be a scalar (in which case it is assumed for
%   all variables).  if [], default to 0.
% <sigma> is the desired covariance matrix (which should be positive definite).
%   the interpretation should be that this is the covariance matrix after
%   each variable has been standardized (i.e. mean-subtracted and
%   normalized to standard deviation 1).  thus, there should be 1s along
%   the diagonal of <sigma>.  notice that the correlation 
%   coefficient is just the dot product of standardized variables, divided
%   by n.  thus, a <sigma> of [1 .5; .5 1] corresponds to the case of
%   two standardized variables with a correlation coefficient of .5.
% <sd> (optional) is a vector with the desired standard deviation for each variable.
%   can be a scalar (in which case it is assumed for all variables).
%   default: 1.
%
% randomly draw <n> points from a multivariate Gaussian defined by
% <mn>, <sigma>, and <sd>.  return a matrix of size <n> x size(sigma,1).
%
% note that we first generate random numbers consistent with <sigma>,
% and then do scaling (<sd>) and adding (<mn>) if necessary.

% calc
dim = size(sigma,1);

% input
if isempty(mn)
  mn = 0;
end
if ~exist('sd','var') || isempty(sd)
  sd = 1;
end

% cholesky decomposition, compute R
if all(sigma(:)==1)  % special case
  R = zeros(size(sigma));
  R(1,:) = 1;
else
  [R,p] = chol(sigma);
  assert(p==0,'sigma is not positive definite');
end

% do it
f = randn(n,dim)*R;  % aha, we just mix the random numbers together in certain proportions
  % let X be randn(n,dim).  then, f = X*R.
  % the covariance of f is (X*R)'*(X*R) / n = R'*X'*X*R / n = R'*R*(n*I) / n = sigma.
  % thus, the covariance of f at this point is the desired sigma.
  % moreover, at this point, each variable is standardized (mean-subtracted, standard deviation 1).
  % to achieve different standard deviations, we just need to scale each variable, as we do below.

% scale if necessary (standard deviation)
if ~all(sd==1)
  if isscalar(sd)
    f = f * sd;
  else
    f = f .* repmat(sd,[n 1]);
  end
end

% add constant if necessary (mean)
if ~all(mn==0)
  if isscalar(mn)
    f = f + mn;
  else
    f = f + repmat(mn,[n 1]);
  end
end
