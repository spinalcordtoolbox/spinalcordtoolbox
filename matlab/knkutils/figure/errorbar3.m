function f = errorbar3(x,y,er,direction,color)

% function f = errorbar3(x,y,er,direction,color)
%
% <x>,<y>,<er> are row vectors of the same length
%   if <er> has imaginary numbers (determined via any(imag(er(:))~=0)),
%   then real(<er>) and imag(<er>) determines the downward and 
%   upward (or leftward and rightward) extent of the error bars,
%   respectively.  <er> can also be 2 x N where the first row
%   has values for the lower bound and the second row has values
%   for the upper bound.
% <direction> is
%   0 or 'h' or 'x' means error on x.  in this case, <y> should be sorted.
%   1 or 'v' or 'y' means error on y.  in this case, <x> should be sorted.
% <color> is a color (e.g. 'r', [1 .5 1])
%
% draw error polygon on the current figure, returning the handle.
% note that the order of <x> (and <y> and <er>) determines the order
% of the vertices of the polygon.
%
% it is okay if there are NaNs in <x>,<y>,<er> --- if there are,
% we propagate the NaNs across <x>,<y>,<er>, and then we draw a 
% separate polygon for each run of non-NaN numbers that consists 
% of at least two elements.  in this case, we may potentially
% return a vector of handles.
%
% example:
% figure; errorbar3(1:10,randn(1,10),abs(randn(1,10))/4,1,[1 .2 .4]);

% propagate NaNs
bad = isnan(x) | isnan(y) | any(isnan(er),1);
x(bad) = NaN;
y(bad) = NaN;
er(:,bad) = NaN;

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
isimag = any(imag(er(:))~=0);

% find good runs
runs = splitruns(x);

% hold on
prev = ishold;
hold on;

% do it
f = [];
for p=1:length(runs)
  if length(runs{p}) > 1
    if isimag
      switch direction
      case {0 'h' 'x'}
        f(end+1) = patch([x(runs{p})-real(er(runs{p})) fliplr(x(runs{p})+imag(er(runs{p})))],[y(runs{p}) fliplr(y(runs{p}))],color);
      case {1 'v' 'y'}
        f(end+1) = patch([x(runs{p}) fliplr(x(runs{p}))],[y(runs{p})-real(er(runs{p})) fliplr(y(runs{p})+imag(er(runs{p})))],color);
      otherwise
        error('invalid <direction>');
      end
    else
      switch direction
      case {0 'h' 'x'}
        f(end+1) = patch([x(runs{p})-er(runs{p}) fliplr(x(runs{p})+er(runs{p}))],[y(runs{p}) fliplr(y(runs{p}))],color);
      case {1 'v' 'y'}
        f(end+1) = patch([x(runs{p}) fliplr(x(runs{p}))],[y(runs{p})-er(runs{p}) fliplr(y(runs{p})+er(runs{p}))],color);
      otherwise
        error('invalid <direction>');
      end
    end
  end
end
set(f,'EdgeColor','none');

% hold off
if ~prev
  hold off;
end
