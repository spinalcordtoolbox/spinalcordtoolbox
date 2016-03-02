function f = calcsurfaceareanball(n)

% function f = calcsurfaceareanball(n)
%
% <n> is the number of dimensions
%
% calculate the log of the surface area of a unit n-ball.
% (unit n-ball means an n-dimensional hypersphere with radius 1.)
%
% example:
% allzero(calcsurfaceareanball(2) - log(2*pi))

% non-logged is f = n*cn*radius^(n-1)
f = log(n) + calclogcn(n);

%%%%%

function f = calclogcn(n)

% calculate the log of cn

if mod(n,2)==0
  f = n/2 * log(pi) - sum(log(1:n/2));
else
  f = (n+1)/2 * log(2) + (n-1)/2 * log(pi) - sum(log(1:2:n));
end
