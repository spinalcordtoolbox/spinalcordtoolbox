function f = randomword(n,mode)

% function f = randomword(n,mode)
%
% <n> is the desired length of the word
% <mode> (optional) is
%   0 means choose from capital letters
%   1 means choose from capital and lowercase letters and digits
%   2 means choose from capital and lowercase letters
%   3 means choose from lowercase letters
%   4 means choose from digits
%   default: 0.
%
% return a random word consisting of <n> characters.
%
% example:
% randomword(5)

% input
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end

% do it
switch mode
case 0
  rng = [64+(1:26)];
case 1
  rng = [64+(1:26) 96+(1:26) 47+(1:10)];
case 2
  rng = [64+(1:26) 96+(1:26)];
case 3
  rng = [96+(1:26)];
case 4
  rng = [47+(1:10)];
end
%f = char(rng(randint(1,n,[1 length(rng)])));
f = char(rng(ceil(rand(1,n)*length(rng))));
