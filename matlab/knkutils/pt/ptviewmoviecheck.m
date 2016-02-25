function [keytimes,badtimes,keybuttons] = ptviewmoviecheck(timeframes,timekeys,deltatime,badkey,deltatimeBAD)

% function [keytimes,badtimes,keybuttons] = ptviewmoviecheck(timeframes,timekeys,deltatime,badkey,deltatimeBAD)
% 
% <timeframes>,<timekeys> are outputs from ptviewmovie.m
% <deltatime> (optional) is the time differential in seconds that must
%   be exceeded for a key that is detected twice in a row to be
%   counted as an actual keypress.  default: 1.
% <badkey> (optional) is a character.  separately process entries in <timekeys>
%   for which the first character matches <badkey>.  if [] do nothing special.
%   default: [].
% <deltatimeBAD> (optional) is like <deltatime> but for <badkey>.  default: 0.25.
%
% draw some figures visualizing the contents of <timeframes> and <timekeys>.
% return a vector with keypress times in <keytimes> and a vector with
% badkey times in <badtimes>.  the buttons corresponding to <keytimes> are
% returned in <keybuttons>.
%
% history:
% - 2014/09/16 - now return keybuttons as output
% - 2014/06/12 - revamp to function behavior. now return useful outputs.
%
% example:
% pton;
% [timeframes,timekeys] = ptviewmovie(uint8(255*rand(100,100,3,100)),[],[],2,[],[],[],[],[],0);
% ptoff;
% [keytimes,badtimes] = ptviewmoviecheck(timeframes,timekeys);

% input
if ~exist('deltatime','var') || isempty(deltatime)
  deltatime = 1;
end
if ~exist('badkey','var') || isempty(badkey)
  badkey = [];
end
if ~exist('deltatimeBAD','var') || isempty(deltatimeBAD)
  deltatimeBAD = 0.25;
end

% look at difference in frame times
drawnow; figure; plot(diff(timeframes),'r-');
xlabel('frame difference'); ylabel('duration (seconds)');
title('inspection of timeframes');

% look at timekeys
drawnow; figure; setfigurepos([50 50 800 300]); hold on;

% first we have to expand the multiple-keypresses cases
timekeysB = {};
for p=1:size(timekeys,1)
  if iscell(timekeys{p,2})
    for pp=1:length(timekeys{p,2})
      timekeysB{end+1,1} = timekeys{p,1};
      timekeysB{end,2} = timekeys{p,2}{pp};
    end
  else
    timekeysB(end+1,:) = timekeys(p,:);
  end
end

% now process non-badkeys
oldkey = ''; oldkeytime = -Inf; cnt = 0;
keytimes = [];
keybuttons = {};
xvals = [];
for p=1:size(timekeysB,1)
  if ~isequal(timekeysB{p,2},'absolutetimefor0') && ...
     (isempty(badkey) || ~isequal(timekeysB{p,2}(1),badkey)) && ...
     (timekeysB{p,1}-oldkeytime > deltatime)     %%%REMOVED ~isequal(timekeysB{p,2},oldkey) ||  [this means all the same]
    keytimes = [keytimes timekeysB{p,1}];  % record
    keybuttons = [keybuttons {timekeysB{p,2}}];  % record
    straightline(timekeysB{p,1},'v',[getcolorchar(cnt) '-']);
    text(timekeysB{p,1},1.1 + 0.5*rand,timekeysB{p,2}(1),'HorizontalAlignment','center','Color',getcolorchar(cnt));
    cnt = cnt + 1;
    oldkey = timekeysB{p,2};
    oldkeytime = timekeysB{p,1};
    xvals = [xvals timekeysB{p,1}];
  end
end

% short in the y-direction so the badkeys are short vertical lines
ax = axis;
axis([ax(1:2) 0 0.5]);

% now process badkeys
oldkey = ''; oldkeytime = -Inf; cnt = 0;
badtimes = [];
for p=1:size(timekeysB,1)
  if ~isequal(timekeysB{p,2},'absolutetimefor0') && ...
     (~isempty(badkey) && isequal(timekeysB{p,2}(1),badkey)) && ...
     (timekeysB{p,1}-oldkeytime > deltatimeBAD)  % if we have found a new key
    badtimes = [badtimes timekeysB{p,1}];  % record
    set(straightline(timekeysB{p,1},'v','k-'),'Color',[.7 .7 .7]);
%%    text(timekeysB{p,1},1.1 + 0.5*rand,timekeysB{p,2}(1),'HorizontalAlignment','center','Color',getcolorchar(cnt));
    cnt = cnt + 1;
    oldkey = timekeysB{p,2};
    oldkeytime = timekeysB{p,1};
    xvals = [xvals timekeysB{p,1}];
  end
end

% finish up figure
xlabel('time (seconds)');
axis([floor(min(xvals)-1) ceil(max(xvals)+1) 0 1.6]);
title(sprintf('inspection of timekeys (repeated keys within %.1f s are not shown)',deltatime));
