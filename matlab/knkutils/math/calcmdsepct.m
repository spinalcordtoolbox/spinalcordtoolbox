function [md,pp,mdpp,dist] = calcmdsepct(m,dim,numboot,fun)

% function [md,pp,mdpp,dist] = calcmdsepct(m,dim,numboot,fun)
%
% <m> is a matrix
% <dim> (optional) is the dimension of interest.
%   default to 2 if <m> is a row vector and to 1 if not.
%   special case is 0 which means to calculate globally.
% <numboot> (optional) is number of bootstraps to take.  Default: 1000.
% <fun> (optional) is a function that accepts a matrix and
%   a dimension and calculates something along that dimension, collapsing
%   it to a single entry.  Default: @nanmedian.
%
% return:
%  <md> as the nanmedian of <m>.
%  <pp> as the 15.87th and 84.13th percentiles of the bootstrapped nanmedians.
%     (68% confidence interval).
%  <mdpp> as the concatenation of <md> and <pp> along <dim>.
%  <dist> as the bootstrapped nanmedians.
%
% note that if <fun> is specified, we use that instead of nanmedians.
% 
% the size of <md> is the same as <m> except collapsed along <dim>.
% the size of <pp> is the same as <m> except having two elements along <dim>.
% the size of <dist> is the same as <m> except having <numboot> along <dim>.
% in the special case where <dim> is 0, <md> is 1 x 1, <pp> is 2 x 1, 
% <mdpp> is 3 x 1, and <dist> is <numboot> x 1.
%
% example:
% [md,pp] = calcmdsepct(randn(1,10000))

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
if ~exist('fun','var') || isempty(fun)
  fun = @nanmedian;
end

% do it
if dim==0
  m = m(:);
  dim = 1;
end
md = fun(m,dim);
dist = bootstrapdim(m,dim,@(x) fun(x,dim),numboot);
%pp = prctile(dist,[25 75],dim);
pp = prctile(dist,[15.87 84.13],dim);
mdpp = cat(dim,md,pp);
