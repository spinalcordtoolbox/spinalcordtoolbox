function f = signedarraypower(m,pow)

% function f = signedarraypower(m,pow)
%
% <m> is a matrix
% <pow> is an exponent
%
% return sign(m).*(abs(m).^pow).
%
% example:
% isequal(signedarraypower([2 -2],2),[4 -4])

f = sign(m).*(abs(m).^pow);
