function [secs, keyCode, deltaSecs] = KbReleaseWait(deviceNumber, varargin)
% [secs, keyCode, deltaSecs] = KbReleaseWait([deviceNumber][, untilTime=inf][, more optional args for KbWait]);
%
% KbReleaseWait waits until all keys on the keyboard are released.
%
% It also returns if the optional deadline 'untilTime' is reached.
%
% This is a convenience wrapper, doing the same thing as
% KbWait(deviceNumber, 1, ...); so read "help KbWait" for details about
% operation and returned values.
%
% You'll typically use this function to make sure that all keys are idle
% before you start some new trial that collects keyboard responses, after
% you've used KbCheck, KbWait or KbPressWait for collecting a response. A
% different approach is to use KbStrokeWait.
%
% See also: KbPressWait, KbReleaseWait, KbWait, KbCheck, KbStrokeWait.

% History:
% 9.3.2008 Written. (MK)

% Assign default device [] if none specified:
if nargin < 1
    deviceNumber = [];
end

% Just call KbWait in 'forWhat' mode 1, passing along all input args and
% returning all output args:
[secs, keyCode, deltaSecs] = KbWait(deviceNumber, 1, varargin{:});

return;
