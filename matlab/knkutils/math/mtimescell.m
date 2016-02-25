function f = mtimescell(m1,m2)

% function f = mtimescell(m1,m2)
%
% <m1> is A x B
% <m2> is a cell vector of matrices such that cat(1,m2{:}) is B x C
%
% simply return <m1>*cat(1,m2{:}) but do so in a way that doesn't cause 
% too much memory usage.
%
% example:
% x = randn(10,20);
% y = randn(20,200);
% result = x*y;
% result2 = mtimescell(x,splitmatrix(y,1,repmat(2,[1 10])));
% allzero(result-result2)

f = 0;
cnt = 0;
for q=1:length(m2)
  f = f + m1(:,cnt + (1:size(m2{q},1))) * m2{q};
  cnt = cnt + size(m2{q},1);
end
