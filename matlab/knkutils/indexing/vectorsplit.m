function f = vectorsplit(v,n)

% function f = vectorsplit(v,n)
%
% <v> is a vector
% <n> is the desired length of a single section
%
% return a cell vector of vectors such that
% the first <n> elements of <v> are in the first vector,
% the second <n> elements are in the second vector, 
% and so on.
%
% example:
% isequal(vectorsplit(1:10,4),{1:4 5:8 9:10})

% calc
len = length(v);
numsections = ceil(len/n);

% do it
f = {};
for p=1:numsections
  f{p} = v(((p-1)*n + 1):min(len,(p-1)*n + n));
end
