function str = fixstr(str)

% function str = fixstr(str)
%
% <str> is a string
%
% return a string suitable for use with MATLAB's figure functions like
% xlabel, title, etc.  specifically, we replace _ with \_ so that this
% does not get interpreted as a subscript.
%
% example:
% figure;
% xlabel('blah_test');
% ylabel(fixstr('blah_test'));

str = regexprep(str,'_','\\_');
