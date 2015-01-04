function u = minL1Spars( f, gamma )
%MINL1SPARS Solves the sparsity problem
% 
% argmin \| u \|_0 + \| u - f \|_1 -> min
% 
% using thresholding

u = minSpars(f, gamma, p);

end

