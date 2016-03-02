function f = circulardiff(m,ref,modv)

% function f = circulardiff(m,ref,modv)
%
% <m> is a matrix or scalar
% <ref> is a matrix or scalar
% <modv> is the modulus
%
% return something like <m>-<ref> where values are in [-<modv>/2,<modv>/2).
% if <m> and <ref> are both matrices, they should have the same size.
%
% example:
% isequal(circulardiff(4,0.5,6),-2.5)

f = mod(m-ref+modv/2,modv)-modv/2;
