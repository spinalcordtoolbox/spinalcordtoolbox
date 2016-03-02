function allocatememory(x)

% function allocatememory(x)
% 
% <x> is the number of gigabytes of memory to allocate
%
% allocate <x> GBs of memory and then return.
% the point of this function is to force swapping.

blah = zeros(12500,round(1000 * x*1000/100));
