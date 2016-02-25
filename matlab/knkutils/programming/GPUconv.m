function x = GPUconv(x,flag)

% function x = GPUconv(x,flag)
%
% <x> is a matrix
% <flag> is
%   0 means do nothing
%   1 means convert to GPUsingle format
%   -1 means convert to single format
%
% example:
% isa(GPUconv([1 2 3],1),'GPUsingle')

switch flag
case 0
case 1
  x = GPUsingle(x);
case -1
  x = single(x);
end
