function f = mat2cellstr(m)

% function f = mat2cellstr(m)
%
% <m> is a matrix of numbers
%
% return a cell matrix of strings.  note that we use num2str 
% to do the conversion, and it does apply some rounding 
% for display purposes.
%
% example:
% isequal(mat2cellstr([1 2 3]),{'1' '2' '3'})

f = cell(size(m));
for p=1:numel(m)
  f{p} = num2str(m(p));
end
