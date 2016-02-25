function f = linspacefixeddiff(x,d,n)

% function f = linspacefixeddiff(x,d,n)
%
% <x> is a number
% <d> is difference between successive numbers
% <n> is the number of desired points (positive integer)
%
% return a vector of equally spaced values starting at <x>.
%
% example:
% isequal(linspacefixeddiff(0,2,5),[0 2 4 6 8])

x2 = x+d*(n-1);
f = linspace(x,x2,n);
