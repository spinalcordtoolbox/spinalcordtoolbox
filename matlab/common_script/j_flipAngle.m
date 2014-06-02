% =========================================================================
% FUNCTION
% j_flipAngle.m
%
% Calculate optimal flip angle
%
% INPUT
% tr                in ms
%
% OUTPUT
% fa                in grad
%
% COMMENTS
% Julien Cohen-Adad 2007-01-09
% =========================================================================
function varargout = j_flipAngle(tr)


% init
if (nargin<1) help j_flipAngle; return; end
t1 = 1000; % in ms (default value in grey matter is 1000 ms)


% perform fa
fa = acos(exp(-tr/t1))*180/pi;


% output
varargout{1} = fa;
