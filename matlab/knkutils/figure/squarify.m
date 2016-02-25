function [rows,cols,f] = squarify(x)

% function [rows,cols,f] = squarify(x)
%
% <x> is a positive integer
%
% given that there are <x> things, return the number of rows <rows>
% and the number of columns <cols> that achieve roughly square dimensions
% (see code for details). <f> is simply [<rows> <cols>].
% note that <rows>*<cols> >= <x>.
%
% example:
% [rows,cols] = squarify(15);
% isequal(rows,4) & isequal(cols,4)

rows = ceil(sqrt(x));
cols = ceil(x/rows);
f = [rows cols];
