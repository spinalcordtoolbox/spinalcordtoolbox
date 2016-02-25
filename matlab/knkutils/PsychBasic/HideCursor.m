function HideCursor(screenid, mouseid)
% HideCursor([screenid=0][, mouseid]
% 
% HideCursor hides the mouse cursor associated with screen 'screenid'.
% By default, the cursor of screen zero on Linux, and all screens on
% Windows and Mac OS/X is hidden. 'mouseid' defines which of multiple
% cursors shall be hidden on Linux. The parameter is silently ignored
% on other systems.
% _________________________________________________________________________
%
% See ShowCursor, SetMouse

% 7/23/97  dgp Added wish.
% 8/15/97  dgp Explain hide/show counter.
% 3/27/99  dgp Mention Backgrounding.
% 3/28/99  dgp Show how to turn off backgrounding. 
% 1/22/00  dgp Cosmetic.
% 4/25/02  dgp Mention conflict with QuickDEX.
% 4/14/03  awi ****** OS X-specific fork from the OS 9 version *******
%               Added call to Screen('HideCursor'...) for OS X.
% 7/12/04  awi Cosmetic and uses IsMac.
% 11/16/04 awi Renamed "HideCursor" to "HideCursorHelper"

%on OS X the Screen.mexmac hides the cursor, not 
%HideCursor.mexmac.  HideCursor.m wraps the 
%Screen call to emulate HideCursor.mex

if nargin < 1
  screenid = 0;
end

if isempty(screenid)
  screenid = 0;
end

if nargin < 2
  mouseid = [];
end

Screen('HideCursorHelper', screenid, mouseid);
