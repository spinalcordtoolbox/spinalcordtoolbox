function [secs, keyCode, deltaSecs] = KbPressWait(deviceNumber, varargin)
% [secs, keyCode, deltaSecs] = KbPressWait([deviceNumber][, untilTime=inf][, more optional args for KbWait]);
%
% KbPressWait waits for a single key press of your subject, ie. it waits
% until all keys on the keyboard are released, after that it waits for a
% a press of a key, then it returns the keyboard state and timestamp of the key
% press *without* waiting for the key press to finish ie, without waiting for
% a key release.
%
% It also returns if the optional deadline 'untilTime' is reached.
%
% This is a convenience wrapper, doing the same thing as
% KbWait(deviceNumber, 2, ...); so read "help KbWait" for details about
% operation and returned values.
%
% You'll typically use this function to ask your subject for a response and
% you want to continue your script immediately without delay, even if the
% subject keeps the key pressed for a while -- a typical usage would be if
% you want to change the stimulus immediately after response, eg, blank the
% display etc.
%
% See also: KbPressWait, KbReleaseWait, KbWait, KbCheck, KbStrokeWait.

% History:
% 9.3.2008 Written. (MK)

% Assign default device [] if none specified:
if nargin < 1
    deviceNumber = [];
end

% Just call KbWait in 'forWhat' mode 2, passing along all input args and
% returning all output args:
[secs, keyCode, deltaSecs] = KbWait(deviceNumber, 2, varargin{:});

return;
