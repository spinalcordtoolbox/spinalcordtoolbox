function f = gettimestring

% function f = gettimestring
%
% return a string with the date and time.  for example, '20140427202543' means that 
% the function was called at 2014/04/27 at 8:25:43pm.
%
% note that we round the seconds down.
%
% example:
% gettimestring

clock0 = fix(clock);
f = sprintf('%04d%02d%02d%02d%02d%02d',clock0(1),clock0(2),clock0(3),clock0(4),clock0(5),clock0(6));
