function evalme

% function evalme
%
% accept multi-line input from stdin (input is terminated by a single '.'
% on a new line).  then evaluate the input in the 'caller' workspace.
%
% this function is useful for copy-and-pasting a large chunk of code
% to be evaluated in the MATLAB workspace.
%
% example:
% evalme
% a=0;
% a+1
% .

evalin('caller',inputmulti);
