function out = EnforcePos(out)
%  out = EnforcePos(in)
%
% Enforce the constraint that spectral power must be
% positive. 
%
% A sophisticated algorithm might do something with small
% positive values as well.
%
% 10/23/93  dhb  Wrote it.

out(out<0) = 0;

