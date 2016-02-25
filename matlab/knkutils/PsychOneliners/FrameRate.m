function hz=FrameRate(windowOrScreenNumber)
% hz=FrameRate([windowOrScreenNumber])
%
% Returns accurate frame rate of window or screen. Default is the main
% screen. The windowOrScreenNumber and frame rate are cached so subsequent
% calls after the first take no time at all.
% See also "Screen GetFlipInterval?"

% 11/8/06 dgp Wrote it.

persistent hzInFrameRate
persistent wInFrameRate
if nargin<1
    windowOrScreenNumber=0;
end
AssertOpenGL;
if ismember(windowOrScreenNumber,wInFrameRate)
    % Use cached frame rate.
    [ok,n]=ismember(windowOrScreenNumber,wInFrameRate);
    hz=hzInFrameRate(n);
    return
end
% Not in cache, so measure it.
if ismember(windowOrScreenNumber,Screen('Screens'))
    w=Screen('OpenWindow',windowOrScreenNumber);
    hz=1/Screen('GetFlipInterval',w);
    Screen('Close',w);
elseif ismember(windowOrScreenNumber,Screen('Windows'))
    hz=1/Screen('GetFlipInterval',windowOrScreenNumber);
else
    error('Illegal windowOrScreenNumber value %.0f',windowOrScreenNumber);
end
% Add it to cache.
wInFrameRate(end+1)=windowOrScreenNumber;
hzInFrameRate(end+1)=hz;

