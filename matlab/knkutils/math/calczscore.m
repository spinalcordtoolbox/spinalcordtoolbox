function [f,mn,sd] = calczscore(m,dim,mn,sd,wantcaution)

% function [f,mn,sd] = calczscore(m,dim,mn,sd,wantcaution)
%
% <m> is a matrix
% <dim> (optional) is the dimension of interest.
%   if supplied, z-score each case oriented along <dim>.
%   if [] or not supplied, z-score globally.
% <mn>,<sd> (optional) is a special case.  supply both of these
%   and we will use them instead of calculating the actual
%   mean and std dev.  to indicate not-supplied, pass in [].
% <wantcaution> (optional) is whether to try to detect weird cases (see zerodiv.m).
%   default: 1.
%
% z-score <m>, operating on individual cases or globally.
% we use nanmean and nanstd to deal with NaNs gracefully.
% also return the mean and std dev matrices.
%
% note that if the nanstd is 0, we return NaNs for that case.
% note that if <wantcaution>, then if the nanstd is close to 0
%   (abs value < 1e-5), we issue a warning (see zerodiv.m),
%   and act as if it is exactly 0, and return NaNs for that case.
%
% note some weird cases:
%   calczscore([]) is []
%   calczscore([NaN NaN]) is [NaN NaN]
%
% example:
% a = [1 2 3];
% isequal(calczscore(a),[-1 0 1])

% inputs
if ~exist('wantcaution','var') || isempty(wantcaution)
  wantcaution = 1;
end

% do it
if ~exist('dim','var') || isempty(dim)
  if ~exist('mn','var') || isempty(mn)
    mn = nanmean(m(:));
    sd = nanstd(m(:));
  end
  f = zerodiv(m-mn,sd,NaN,wantcaution);
else
  if ~exist('mn','var') || isempty(mn)
    mn = nanmean(m,dim);
    sd = nanstd(m,0,dim);
  end
  f = bsxfun(@(x,y) zerodiv(x,y,NaN,wantcaution),bsxfun(@minus,m,mn),sd);
end
