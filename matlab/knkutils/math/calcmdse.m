function f = calcmdse(m,dim,numboot,wantmn)

% function f = calcmdse(m,dim,numboot,wantmn)
%
% <m> is a matrix
% <dim> (optional) is the dimension of interest.
%   default to 2 if <m> is a row vector and to 1 if not.
%   special case is 0 which means to calculate globally.
% <numboot> (optional) is number of bootstraps to take.  default: 1000.
% <wantmn> (optional) is whether to use nanmean instead.  default: 0.
%
% return A+j*B where A is the nanmedian of <m> and B is the standard error 
% on the nanmedian of <m> computed via bootstrapping.  the size of the result
% is the same as <m> except collapsed along <dim>.  (in the special case
% where <dim> is 0, the result is a scalar.)
%
% note that we quantify the standard error by taking the standard deviation
% across bootstraps.  this may be inaccurate for non-Gaussian distributions.
% TODO: implement percentiles.
%
% history:
% 2011/07/01 - now use nanmedian and nanmean!
%
% example:
% calcmdse(randn(1,10000))

% input
if ~exist('dim','var') || isempty(dim)
  if isvector(m) && size(m,1)==1
    dim = 2;
  else
    dim = 1;
  end
end
if ~exist('numboot','var') || isempty(numboot)
  numboot = 1000;
end
if ~exist('wantmn','var') || isempty(wantmn)
  wantmn = 0;
end

% do it
if dim==0
  m = m(:);
  dim = 1;
end
if wantmn
  f = nanmean(m,dim) + j*std(bootstrapdim(m,dim,@(x) nanmean(x,dim),numboot),[],dim);
else
  f = nanmedian(m,dim) + j*std(bootstrapdim(m,dim,@(x) nanmedian(x,dim),numboot),[],dim);
end
