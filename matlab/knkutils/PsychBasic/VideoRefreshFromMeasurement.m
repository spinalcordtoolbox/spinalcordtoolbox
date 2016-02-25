function ifi = VideoRefreshFromMeasurement(window, samples)
% ifi = VideoRefreshFromMeasurement(window [, samples=600]);
%
% Measure video refresh interval for onscreen windows display with a method
% that is supposed to be as robust and accurate as possible.
%
% 'window' is the onscreen window handle for the window whose corresponding
% display should be measured.
%
% 'samples' is the (optional) number of refresh samples to take into
% account.
%
% This routine returns the estimated refresh duration in seconds. It uses
% one of multiple calibration strategies, preferring more accurate ones but
% falling back to less accurate ones if the accurate ones are not supported
% by your system:
%
% 1. On MacOS/X it tries to use low-level VBL interrupt timestamps.
% 2. If that fails it tries to get the measurement from the beamposition
%    measurement method.
% 3. If that fails it returns the measurment from
%    Screen('GetFlipInterval').
%

% History:
% 07/04/07 Written (MK).

if nargin < 1
    error('You must provide a ''window'' handle for calibration!');
end

if Screen('WindowKind', window) ~= 1
    error('You must provide an onscreen window handle for calibration!');
end

if nargin < 2
    samples = 600;
end

wininfo = Screen('GetWindowInfo', window);
startvblcount = wininfo.VBLCount;

if startvblcount > 0
    % VBL counter supported by OS. Use it for measurement:
    newvblcount = startvblcount;
    while newvblcount == startvblcount
        wininfo = Screen('GetWindowInfo', window);
        newvblcount = wininfo.VBLCount;
    end

    startvblcount = newvblcount;
    tstart = wininfo.LastVBLTime;

    while newvblcount < startvblcount + samples
        wininfo = Screen('GetWindowInfo', window);
        newvblcount = wininfo.VBLCount;
    end

    vblcount = newvblcount - startvblcount;
    telapsed = wininfo.LastVBLTime - tstart;

    ifi = telapsed / vblcount;
else
    % No VBL counter support :( - Try beamposition method:
    ifi = wininfo.VideoRefreshFromBeamposition;
    
    % Result from beamposition method?
    if ifi == 0
        % Nope. Ok we return the flip interval measured by Screen:
        ifi = Screen('GetFlipInterval', window);
    end
end

% We should have an ifi, one or the other way...
return;
