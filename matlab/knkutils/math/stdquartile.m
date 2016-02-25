function f = stdquartile(m,dim,mode)

% function f = stdquartile(m,dim,mode)
%
% <m> is a matrix
% <dim> (optional) is the dimension of interest.
%   default to 2 if <m> is a row vector and to 1 if not.
%   special case is 0 which means to calculate globally.
% <mode> (optional) is
%   0 means [25 50 70]
%   1 means [16 50 84] (68% confidence interval)
%   2 means [2.5 50 97.5] (95% confidence interval)
%   default: 0.
%
% return A+j*B where A is p50-p25 and B is p75-p50 (where pN means
% the Nth percentile). (other percentiles can be obtained by 
% setting <mode> appropriately.)  the size of the result is the same 
% as <m> except collapsed along <dim>.  (in the special case where <dim> 
% is 0, the result is a scalar.)
%
% example:
% x = randn(1,1000);
% figure; hold on;
% hist(x);
% ax = axis;
% errorbar2(median(x),.9*ax(4),stdquartile(x),'h','r-');

% input
if ~exist('dim','var') || isempty(dim)
  if isvector(m) && size(m,1)==1
    dim = 2;
  else
    dim = 1;
  end
end
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end

% calc
switch mode
case 0
  pp = [25 50 75];
case 1
  pp = [50-34 50 50+34];
case 2
  pp = [2.5 50 97.5];
end

% do it
if dim==0
  m = m(:);
  dim = 1;
end
flow = prctile(m,pp(1),dim);
fmid = prctile(m,pp(2),dim);
fhigh = prctile(m,pp(3),dim);
f = fmid-flow + j*(fhigh-fmid);
