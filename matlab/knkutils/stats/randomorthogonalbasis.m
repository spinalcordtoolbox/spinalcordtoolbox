function f = randomorthogonalbasis(n1,n2)

% function f = randomorthogonalbasis(n1,n2)
%
% <n1> is the dimensionality
% <n2> is the number of basis functions desired
%
% return a matrix of dimensions <n1> x <n2>.
% the columns are randomly-generated unit-length orthogonal basis functions.
%
% example:
% allzero(calcconfusionmatrix(randomorthogonalbasis(10,10),[],0) - eye(10))

f = unitlength(randn(n1,1));
for p=2:n2
  f = cat(2,f,unitlength(projectionmatrix(f)*randn(n1,1)));
end
