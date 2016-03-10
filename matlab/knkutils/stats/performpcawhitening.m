function [f,trans,ev] = performpcawhitening(x,mode)

% function [f,trans,ev] = performpcawhitening(x,mode)
%
% <x> is a matrix (points x dimensions)
% <mode> (optional) is
%   0 is scale
%   1 is scale and rotate
%   2 is rotate
%   default: 0.
%
% perform PCA whitening and then return the result (same dimensions as <x>).
% also return the transformation matrix that we used.
% also return a vector with the eigenvalues.
%
% example:
% a = generatepinknoise(20,[],1000,1);
% b = permute(reshape(performpcawhitening(reshape(a,20*20,[])'),[],20,20),[2 3 1]);
% figure; imagesc(a(:,:,1)); axis equal tight;
% figure; imagesc(b(:,:,1)); axis equal tight;

% input
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end

% do it
[u,s,v] = svd(x'*x,0);
ev = flatten(sqrt(diag(s)));
switch mode
case 0
  trans = v*diag(1./ev)*v';
case 1
  trans = v*diag(1./ev);
case 2
  trans = v;
end
f = x*trans;
