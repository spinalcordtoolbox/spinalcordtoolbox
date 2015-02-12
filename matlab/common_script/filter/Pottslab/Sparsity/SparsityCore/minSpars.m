function [u, dataError, nSpikes, energy] = minSpars( f, gamma, p )
%MINSPARS Solves the sparsity problem
% 
%  \| u \|_0 + \| u - f \|_p^p -> min
% 
% using thresholding

u = (abs(f).^p >= gamma) .* f;
dataError = sum(u(:) - f(:)).^p;
nSpikes = sum(u(:) ~= 0);
energy = gamma * nSpikes + dataError;

end

