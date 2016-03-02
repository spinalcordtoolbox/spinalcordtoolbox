function f = getextension(m)

% function f = getextension(m)
%
% <m> is a string referring to a file
%
% return the extension like '.txt'.
% the extension can be empty ('').
%
% example:
% getextension('blah.png')
% getextension('blah')
% getextension('blah.')

[a,b,f] = fileparts(m);
