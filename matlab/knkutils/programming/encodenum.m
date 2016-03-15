function f = encodenum(n)

% function f = encodenum(n)
%
% <n> is an integer or decimal number
%
% convert to a string that consists of all
% uppercase alphabetical characters.  this is
% a deterministic operation, useful for embedding
% numbers into strings that could be in the
% name of MATLAB .m files.
%
% note that because this routine uses num2str,
% long decimals will be truncated.
%
% example:
% encodenum(123.456)

f = char(double(num2str(n))+20);
