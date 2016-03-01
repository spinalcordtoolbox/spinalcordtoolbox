function f = circularavg(m,modv,dim,mode)

% function f = circularavg(m,modv,dim,mode)
%
% <m> is a matrix
% <modv> (optional) is the modulus.  default: 2*pi.
% <dim> (optional) is the dimension of interest.
%   if supplied, compute the circular average along <dim>.
%   if [] or not supplied, compute the circular average of the entire matrix.
% <mode> (optional) is
%   0 means compute the average (of each coordinate)
%   1 means compute the median (of each coordinate)
%
% return the circular average (or median) of <m> along <dim>.  the resulting values 
% will be in the range [0,<modv>).  NaNs in <m> are okay (we just ignore them).
%
% example:
% circularavg([.5 2*pi-.5])

% input
if ~exist('modv','var') || isempty(modv)
  modv = 2*pi;
end
if ~exist('dim','var') || isempty(dim)
  m = m(:);
  dim = 1;
end
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end

% do it
switch mode
case 0
  f = mod(angle(   nansum(ang2complex(m*((2*pi)/modv)),dim)),2*pi)*(modv/(2*pi));
case 1
  f = mod(angle(nanmedian(ang2complex(m*((2*pi)/modv)),dim)),2*pi)*(modv/(2*pi));
end
