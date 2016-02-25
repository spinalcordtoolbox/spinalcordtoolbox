function f = countinstances(m)

% function f = countinstances(m)
%
% <m> is a matrix with positive integers
%
% return a 1 x N vector where N is max(m(:)).
% the Ith element gives the number of occurrences of I in <m>.
%
% example:
% isequal(countinstances([1 2 3 2 1 1]),[3 2 1])

num = max(m(:));
f = zeros(1,num);
for p=1:num
  f(p) = sum(m(:)==p);
end
