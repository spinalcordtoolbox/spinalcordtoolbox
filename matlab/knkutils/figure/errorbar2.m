function f = errorbar2(x,y,er,direction,varargin)

% function f = errorbar2(x,y,er,direction,varargin)
%
% <x>,<y>,<er> are row vectors of the same length
%   if <er> has imaginary numbers (determined via any(imag(<er>)~=0)),
%   then real(<er>) and imag(<er>) determines the downward and 
%   upward (or leftward and rightward) extent of the error bars,
%   respectively.  <er> can also be 2 x N where the first row
%   has values for the lower bound and the second row has values
%   for the upper bound.
% <direction> is
%   0 or 'h' or 'x' means error on x
%   1 or 'v' or 'y' means error on y
% <varargin> are additional arguments to plot.m (e.g. 'r-')
%
% draws error lines on the current figure, returning
% a vector of handles.
%
% example:
% figure; errorbar2(randn(1,10),randn(1,10),randn(1,10)/4,1,'r-');
%
% TODO: what about the cases of NaNs?  e.g. see errorbar3.m.

% hold on
prev = ishold;
hold on;

% prep er
if size(er,1) == 2
  switch direction
  case {0 'h' 'x'}
    er = x - er(1,:) + j*(er(2,:) - x);
  case {1 'v' 'y'}
    er = y - er(1,:) + j*(er(2,:) - y);
  end
end

% calc
isimag = any(imag(er)~=0);

% do it
f = [];
if isimag
  switch direction
  case {0 'h' 'x'}
    for p=1:numel(x)
      f = [f plot([x(p)-real(er(p)) x(p)+imag(er(p))],[y(p) y(p)],varargin{:})];
    end
  case {1 'v' 'y'}
    for p=1:numel(x)
      f = [f plot([x(p) x(p)],[y(p)-real(er(p)) y(p)+imag(er(p))],varargin{:})];
    end
  otherwise
    error('invalid <direction>');
  end
else
  switch direction
  case {0 'h' 'x'}
    for p=1:numel(x)
      f = [f plot([x(p)-er(p) x(p)+er(p)],[y(p) y(p)],varargin{:})];
    end
  case {1 'v' 'y'}
    for p=1:numel(x)
      f = [f plot([x(p) x(p)],[y(p)-er(p) y(p)+er(p)],varargin{:})];
    end
  otherwise
    error('invalid <direction>');
  end
end

% hold off
if ~prev
  hold off;
end
