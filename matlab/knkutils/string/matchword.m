function f = matchword(str)

% function f = matchword(str)
%
% <str> is a string
% 
% return the first word found in <str>,
% determined by the regular expression '(\S+)'.
%
% example:
% isequal(matchword('  blah '),'blah')

f = firstelc(firstelc(regexp(str,'(\S+)','tokens')));
