function [mn,se] = meanandse(m,dim,flag)

% function [mn,se] = meanandse(m,dim,flag)
%
% <m> is a matrix
% <dim> (optional) is a dimension.
%   default to 2 if <m> is a row vector and to 1 if not.
% <flag> (optional) is
%   0 means normal
%   1 means calculate standard deviation (not standard error)
%   default: 0.
%
% return the mean and standard error of <m> along dimension <dim>.
% 
% example:
% [mn,se] = meanandse(randn(1,100))

% input
if ~exist('dim','var') || isempty(dim)
  if isrowvector(m)
    dim = 2;
  else
    dim = 1;
  end
end
if ~exist('flag','var') || isempty(flag)
  flag = 0;
end

% do it
mn = mean(m,dim);
switch flag
case 0
  se = std(m,0,dim)/sqrt(size(m,dim));
case 1
  se = std(m,0,dim);
end
