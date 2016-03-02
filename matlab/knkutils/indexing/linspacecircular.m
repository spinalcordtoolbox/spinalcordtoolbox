function f = linspacecircular(x1,x2,n)

% function f = linspacecircular(x1,x2,n)
%
% <x1>,<x2> are numbers
% <n> is the number of desired points
%
% return a vector of equally spaced values starting at <x1>
% and stopping just before <x2> (<x2> is treated as equivalent
% to <x1>).
%
% example:
% isequal(linspacecircular(0,8,4),[0 2 4 6])

dif = (x2-x1)/n;
f = linspace(x1,x2-dif,n);
