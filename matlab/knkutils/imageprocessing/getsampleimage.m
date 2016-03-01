function f = getsampleimage(n)

% function f = getsampleimage(n)
%
% <n> (optional) is an integer between 1 and 2.  default: 1.
%
% return a sample grayscale natural image (500 x 500) with values in [0,1].
% not gamma-corrected.

% input
if ~exist('n','var') || isempty(n)
  n = 1;
end

% do it
f = double(imread(strrep(which('getsampleimage'),'getsampleimage.m',sprintf('getsampleimage%d.png',n))))/255;
