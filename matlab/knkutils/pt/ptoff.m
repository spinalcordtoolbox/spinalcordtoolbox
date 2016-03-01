function ptoff(oldclut)

% function ptoff(oldclut)
%
% <oldclut> (optional) is the clut to restore.
%   default is [] which means do not restore a clut.
%
% uninitialize the PsychToolbox setup by restoring the clut
% and closing all PsychToolbox windows.
%
% use in conjunction with pton.m.
%
% example:
% pton;
% ptoff;

% input
if ~exist('oldclut','var') || isempty(oldclut)
  oldclut = [];
end

% do it
win = Screen('Windows');
if ~isempty(oldclut)
  Screen('LoadNormalizedGammaTable',win,oldclut);
end
Screen('Close',win);
Screen('CloseAll');
