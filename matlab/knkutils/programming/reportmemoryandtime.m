function reportmemoryandtime

% function reportmemoryandtime
% 
% prints a message that reports the memory usage of the 
% caller workspace, the current time, and the file and
% line number that this function was called from.
% if called from the workspace, we just say "N/A" for
% the file and line number.
%
% example:
% reportmemoryandtime

% figure out caller file and caller line
a = evalin('caller','dbstack');
if length(a) < 2
  sfile = 'N/A';
  sline = 'N/A';
else
  sfile = a(2).file;
  sline = sprintf('%d',a(2).line);
end

% do it
fprintf('***** memory usage: %.f MB, current time: %s (file: %s, line: %s) *****\n',evalin('caller','checkmemoryworkspace'),datestr(now),sfile,sline);




%OLD: warning(sprintf('\n\n***** memory usage: %.f MB, current time: %s *****\n',evalin('caller','checkmemoryworkspace'),datestr(now)));
