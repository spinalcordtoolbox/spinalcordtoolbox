function [u, dataError, nSpikes, energy] = minL2Spars( f, gamma)
%MINL2SPARS Solves the sparsity problem
% 
% argmin \| u \|_0 + \| u - f \|_2^2
% 
% using thresholding

[u, dataError, nSpikes, energy] = minSpars(f, gamma, 2);

end

