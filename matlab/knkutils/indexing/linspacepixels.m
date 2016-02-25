function f = linspacepixels(x1,x2,n)

% function f = linspacepixels(x1,x2,n)
%
% <x1>,<x2> are numbers
% <n> is the number of desired points
%
% return a vector of equally spaced points that can 
% be treated as centers of pixels whose total field-of-view
% would be bounded by <x1> and <x2>.
%
% example:
% isequal(linspacepixels(0,1,2),[.25 .75])

dif = ((x2-x1)/n)/2;  % half the difference between successive points
f = linspace(x1+dif,x2-dif,n);
