function f = roundpvalue(x)

% function f = roundpvalue(x)
%
% <x> is a matrix of p-values
%
% round p-values up such that we can express them in the form Xe-Y.
%
% example:
% allzero(roundpvalue(2.5e-9) - 3e-9)

temp = 10.^-floor(log10(x));  % this is the value you multiply with to get rid of all those zeros
f = ceil(x.*temp)./temp;
