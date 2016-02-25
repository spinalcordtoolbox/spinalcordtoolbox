function f = cell2str(m)

% function f = cell2str(m)
%
% <m> is a 2D cell matrix with elements that are number-matrices or strings.
%
% return the string that can be evaluated to obtain <m>.
%
% example:
% a = {1 [5 6 7]; ':' 'df'};
% isequal(cell2str(a),'{ 1 [5 6 7]; '':'' ''df'';}')
% isequal(a,eval(cell2str(a)))

% do it row by row
f = '{';
for p=1:size(m,1)  % POTENTIALLY SLOW
  for q=1:size(m,2)
    f = [f ' ' mat2str(m{p,q})];
  end
  f = [f ';'];
end
f = [f '}'];
