function f = getcolorchar(x)

% function f = getcolorchar(x)
%
% <x> is an integer
%
% return a color char from {'b' 'g' 'r' 'c' 'm' 'y' 'k'}
% wrapping around as necessary using mod2.m.
%
% example:
% isequal(getcolorchar(1),'b')

colors = {'b' 'g' 'r' 'c' 'm' 'y' 'k'};
f = colors{mod2(x,length(colors))};
