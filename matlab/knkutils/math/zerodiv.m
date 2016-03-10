function f = zerodiv(x,y,val,wantcaution)

% function f = zerodiv(x,y,val,wantcaution)
% 
% <x>,<y> are matrices of the same size or either or both can be scalars.
% <val> (optional) is the value to use when <y> is 0.  default: 0.
% <wantcaution> (optional) is whether to perform special handling of weird
%   cases (see below).  default: 1.
%
% calculate x./y but use <val> when y is 0.
% if <wantcaution>, then if the absolute value of one or more elements of y is 
%   less than 1e-5 (but not exactly 0), we issue a warning and then treat these 
%   elements as if they are exactly 0.
% if not <wantcaution>, then we do nothing special.
%
% note some weird cases:
%   if either x or y is [], we return [].
%   NaNs in x and y are handled in the usual way.
%
% history:
% 2011/02/02 - in the case that y is not a scalar and wantcaution is set to 0, 
%              we were allowing division by 0 to result in Inf and *not* replaced 
%              with val as desired.  big mistake.  we have now fixed this.
%
% example:
% isequalwithequalnans(zerodiv([1 2 3],[1 0 NaN]),[1 0 NaN])

% input
if nargin < 4  % need optimal speed so try to bypass in the fully specified case if we can
  if ~exist('val','var') || isempty(val)
    val = 0;
  end
  if ~exist('wantcaution','var') || isempty(wantcaution)
    wantcaution = 1;
  end
else
  if isempty(val)
    val = 0;
  end
  if isempty(wantcaution)
    wantcaution = 1;
  end
end

% handle special case of y being scalar
if isscalar(y)
  if y==0
    f = repmat(val,size(x));
  else
    if wantcaution && abs(y) < 1e-5   % see allzero.m
      warning('abs value of divisor is less than 1e-5. we are treating the divisor as 0.');
      f = repmat(val,size(x));
    else
%REMOVED:
%      if abs(y) < 1e-5 && ~wantcaution
%        warning('abs value of divisor is less than 1e-5. we are treating the divisor as-is.');
%      end
      f = x./y;
    end
  end
else
  % do it
  bad = y==0;
  bad2 = abs(y) < 1e-5;  % see allzero.m
  if wantcaution && any(bad2(:) & ~bad(:))
    warning('abs value of one or more divisors is less than 1e-5. we are treating these divisors as 0.');
  end
%REMOVED:
%  if any(bad2 & ~bad) && ~wantcaution
%    warning('abs value of one or more divisors is less than 1e-5. we are treating the divisors as-is.');
%  end
  if wantcaution
    y(bad2) = 1;
    f = x./y;
    f(bad2) = val;
  else
    y(bad) = 1;
    f = x./y;
    f(bad) = val;
  end
end
