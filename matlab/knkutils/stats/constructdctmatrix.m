function f = constructdctmatrix(n,ks)

% function f = constructdctmatrix(n,ks)
%
% <n> is the number of points
% <ks> is a vector of nonnegative integers (in the range [0,<n>-1])
%
% return a matrix of dimensions <n> x length(<ks>)
% with DCT-II basis functions in the columns.
%
% example:
% figure; imagesc(constructdctmatrix(100,0:3));

% NOTE: this is hard edged.  sinc in the space domain?  use a butter filter?
%    use polynomials instead?

f = [];
temp = linspacecircular(0,1,n)' + 1/(2*n);
for p=1:length(ks)
  f = cat(2,f,cos(ks(p)*pi*temp));
end
