function f = l1unitlength(m,dim,wantcaution)

% function f = l1unitlength(m,dim,wantcaution)
%
% <m> is a matrix
% <dim> (optional) is the dimension of interest.
%   if supplied, normalize each case oriented along <dim> to have L1 unit length.
%   if [] or not supplied, normalize L1 length globally.
% <wantcaution> (optional) is whether to try to detect weird cases (see zerodiv.m)
%   where the length of <m> is very small to start with.  default: 1.
%
% L1-unit-length-normalize <m> via scaling, operating either on individual cases or globally.
% the output has the same dimensions as <m>.
%
% we ignore NaNs gracefully.
%
% note some weird cases:
%   unitlength([]) is [].
%   unitlength([0 0]) is [NaN NaN].
%   unitlength([NaN NaN]) is [NaN NaN].
%
% example:
% a = [3 0 NaN];
% isequalwithequalnans(l1unitlength(a),[1 0 NaN])

% input
if ~exist('dim','var') || isempty(dim)
  dim = [];
end
if ~exist('wantcaution','var') || isempty(wantcaution)
  wantcaution = 1;
end

% figure out L1 vector length
len = l1vectorlength(m,dim);

% do it
f = bsxfun(@(x,y) zerodiv(x,y,NaN,wantcaution),m,len);
