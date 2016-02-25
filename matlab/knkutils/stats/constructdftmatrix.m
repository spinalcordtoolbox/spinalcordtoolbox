function f = constructdftmatrix(n,ks)

% function f = constructdftmatrix(n,ks)
%
% <n> is the number of points
% <ks> is a vector of nonnegative integers indicating numbers of cycles
%   (which should be less than or equal to <n>/2)
%
% return a matrix of dimensions <n> x B with sine and cosine 
% basis functions in the columns.  the basis functions are
% not scaled.  two basis functions are returned for each element
% of <ks>, except for the cases of 0 or <n>/2 where only the cosine
% basis function is returned (in order to avoid basis functions that
% are all zero).
%
% example:
% figure; plot(constructdftmatrix(20,0:1),'o-');

f = [];
temp = linspacecircular(0,1,n)';
for p=1:length(ks)
  if ks(p) ~= 0 && ks(p) ~= n/2
    f = cat(2,f,sin(ks(p) * 2*pi*temp));
  end
  f = cat(2,f,cos(ks(p) * 2*pi*temp));
end
