function f = alwayszero(varargin)

% function f = alwayszero(varargin)
%
% <varargin> is anything
%
% return 0.  this function is useful for the 'OutputFcn' option
% in the Optimization Toolbox.
%
% example:
% isequal(alwayszero(23,[4 2]),0)

f = 0;
