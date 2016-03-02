function playalarm(n,fctr)

% function playalarm(n,fctr)
%
% <n> (optional) is the number of seconds to play the alarm.
%   special case is -1 which means to play until control-C
%   is pressed (which will then stop all execution).  default: 1.
% <fctr> (optional) is the positive scale factor to apply
%   (to change amplitude).  default: 1.
%
% play an alarm.  note that we aren't that concerned about
% precise timing (see code), so be wary.
%
% example:
% playalarm

% input
if ~exist('n','var') || isempty(n)
  n = 1;
end
if ~exist('fctr','var') || isempty(fctr)
  fctr = 1;
end

% special infinite case
if n==-1
  while 1
    Sound(fctr*[sin(1:1000) zeros(1,1000)]);
  end

% regular case
else
  Sound([1]);  % run to get the cache going
  stime = clock;
  while etime(clock,stime) < n
    Sound(fctr*[sin(1:1000) zeros(1,1000)]);
  end
end
