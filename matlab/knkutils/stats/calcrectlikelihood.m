function f = calcrectlikelihood(varargin)

% function f = calcrectlikelihood(c,data)
%
% <c> is directions x dimensions with unit-length directions
% <data> is points x dimensions with points along the unit hypersphere to evaluate the likelihood at
%
% OR
%
% function f = calcrectlikelihood(dots)
%
% <dots> is points x directions with dot product of each patch with each direction
%
% calculate the likelihood of a set of points given a collection of unit-length directions.
% the architecture is rectified half-squared responses.
% return the average log probability value (up to a scale factor).
%
% see also fitrectpdf.m.
%
% example:
% calcrectlikelihood(unitlength(randn(10,50),2),unitlength(randn(1000,50),2))

% constants
expt = 2;

% the first case
if length(varargin)==2
  c = varargin{1};
  data = varargin{2};

  % calc
  numdirs = size(c,1);
  numpoints = size(data,1);

  % calc likelihood
  f = mean(log2(sum(posrect(data*c').^expt,2)/numdirs + eps));

else
  dots = varargin{1};
  
  % calc
  numdirs = size(dots,2);

  % calc likelihood
  f = mean(log2(sum(posrect(dots).^expt,2)/numdirs + eps));

end
