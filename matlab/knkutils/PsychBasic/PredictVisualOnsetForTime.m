function tonset = PredictVisualOnsetForTime(window, when, refreshinterval)
% tonset = PredictVisualOnsetForTime(window, when [, refreshinterval])
%
% Map a specific requested 'when' visual onset time, as you would pass it as
% 'when' parameter to Screen('Flip', window, when); to the estimated onset
% time of the "flipped" stimulus.
%
% By default, the refresh interval from Screen('GetFlipInterval', window);
% is used for calculation, but you can provide an optional
% 'refreshinterval' to override this choice.
%
% This function predicts the real onset time of your "flipped" stimulus,
% taking into account that Screen('Flip') will not show your stimulus at
% exactly the requested 'when' time, but it will synchronize stimulus onset
% to the display refresh cycle of your monitor, ie, it will wait for onset
% of the closest vertical blanking interval equal or later than 'when'.
%
% You can use the predicted 'tonset' to synchronize other modalities to
% visual stimulus onset. E.g., our sound driver PsychPortAudio can accept
% such a time as sound onset deadline in order to synch sound onset with
% visual stimulus onset...
%
% Of course if your stimulus is too complex to be finished with drawing
% until 'when' then Screen('Flip') will miss the deadline and this
% prediction will be wrong.
%
% Accuracy of prediction:
%
% On systems that support timing via beamposition queries, the prediction
% should be accurate to better than 200 microseconds. On systems without
% beamposition queries, the prediction may be off by up to a millisecond
% worst case. On such systems we can't determine and correct for the
% duration of the vertical blanking interval, so the prediction will be a
% bit too early.

% History:
% 07/03/2007 Written (MK).

if nargin < 2
    error('You must provide a "window" handle and a "when" target time!');
end

% Retrieve last known good VBL timestamp for onscreen window 'window':
wininfo = Screen('GetWindowInfo', window);
lastvbl = wininfo.LastVBLTime;

if lastvbl < 0
    error('Sorry, can''t predict onset time, no valid previous flip timestamp.\nCall Screen(''Flip'') at least once before using this function!');
end

if nargin < 3
    refreshinterval = [];
end

if isempty(refreshinterval)
    % Retrieve measured monitor flip interval:
    ifi = Screen('GetFlipInterval', window);
else
    % Use provided refresh interval instead:
    ifi = refreshinterval;
end

% Need to find first VBL after 'when':
if when <= 0
    % Special case: Swap at next VSYNC:
    % Convert into something we can handle...
    when = lastvbl + 0.5 * ifi;
end

% Compute time delta in units of video refresh intervals between 'when' and
% the last known flip:
dt = (when - lastvbl) / ifi;

% Round 'dt' up to the closest integer greater than 'dt': This will be the
% index of the next refresh interval after 'when'. We add a small epsilon
% to make sure we don't get stuck at 'dt' itself if it should be too close
% to a vbl onset. In that case we would miss that deadline anyway...
dt = floor(dt) + 1;

% Compute corresponding vbl time for target flip video frame 'dt':
vbl = lastvbl + (dt * ifi);

% Ok, vbl is our best guess about VBL onset - and therefore bufferswap time
% of Screen('Flip', window, when);

% If this system supports beamposition queries then we can estimate the
% real visual onset time from the 'vbl' time by adding the duration of the
% VBL. If the system doesnt support them, then 'vbl' is our best estimate
% of onset...
if wininfo.VBLEndline > 0
    % Known VBL endline! We can calculate VBL duration and add it to our
    % 'vbl' estimate.
    vblduration = (wininfo.VBLEndline - wininfo.VBLStartline) / wininfo.VBLEndline * ifi;
    tonset = vbl + vblduration;
else
    % No knowledge about duration of VBL. We return 'vbl' as our best
    % estimate:
    tonset = vbl;
end

return;


