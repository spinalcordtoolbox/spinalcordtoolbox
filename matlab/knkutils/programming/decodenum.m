function f = decodenum(n)

% function f = decodenum(n)
%
% <n> is the output of encodenum.m
%
% undo the transformation of encodenum.m.
%
% example:
% decodenum(encodenum(123.456))

f = str2double(char(double(n)-20));
