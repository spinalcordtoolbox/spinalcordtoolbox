function f = constructpolynomialmatrix(n,degrees)

% function f = constructpolynomialmatrix(n,degrees)
%
% <n> is the number of points
% <degrees> is a vector of polynomial degrees
%
% return a matrix of dimensions <n> x length(<degrees>)
% with polynomials in the columns.  each column is orthogonalized
% with respect to all of the earlier columns and then made 
% unit-length.  please see the code for the exact details.
% beware of numerical precision issues for high degrees...
%
% history:
% - 2014/07/31 - now, we orthogonalize and make unit length.
%                this changes previous behavior!
%
% example:
% X = constructpolynomialmatrix(100,0:3);
% figure; subplot(1,2,1); imagesc(X); subplot(1,2,2); imagesc(X'*X);

% do it
f = [];
temp = linspace(-1,1,n)';
for p=1:length(degrees)

  % construct polynomial
  polyvector = temp .^ degrees(p);
  
  % orthogonalize with respect to earlier polynomials and make unit length
  polyvector = unitlength(projectionmatrix(f)*polyvector);

  % record
  f = cat(2,f,polyvector);
  
end
