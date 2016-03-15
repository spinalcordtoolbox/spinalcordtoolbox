function f = makeletters(n)

% function f = makeletters(n)
%
% <n> is the desired number of letters (between 0 and 26).
%
% return a cell vector of letter characters starting from 'a'.
%
% example:
% isequal(makeletters(3),{'a' 'b' 'c'})

f = mat2cell(char(96 + (1:n)),1,ones(1,n));
