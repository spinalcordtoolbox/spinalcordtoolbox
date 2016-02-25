function ifi = VideoRefreshFromMeasurement(window, samples)
% ifi = VideoRefreshFromMeasurement(window, samples)
%
% This function should determine the exact duration of the displays video
% refresh interval with the highest possible precision, using whatever
% method proves to be the most accurate on your system.
%
% You must provide 'window' the handle to the onscreen window whose
% associated display you want to be measured and (optionally) 'samples',
% the number of samples to take for computation of video refresh interval.
%
% The function returns the measured interval in units of seconds.
%
% CAUTION: This is unfinished alpha quality code. While it works well on
% some system setups, it doesn't yet on others and will need more
% fine-tuning in the future. For most purpose, the values returned by
% Screen('GetFlipInterval', window); are perfectly useable and the
% recommended way of getting the video refresh duration.

% History:
% 07/16/07 Written (MK).

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
    % Empirically this is found to be the most reliable way of determining
    % refresh interavl on Macintosh OS/X:
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
    % Its not yet clear if beamposition method or vbl-synced flip interval
    % is the best way to do it on MS-Windows...
    
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
