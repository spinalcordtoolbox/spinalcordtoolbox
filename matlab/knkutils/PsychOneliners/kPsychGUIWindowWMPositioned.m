function rc = kPsychGUIWindowWMPositioned
% kPsychGUIWindowWMPositioned -- Leave onscreen window placement to system.
%
% This flag can be passed to the optional 'specialFlags' parameter of
% Screen('OpenWindow', ...) or PsychImaging('OpenWindow', ...),
% in addition to the kPsychGUIWindow flag. It has not much of an effect for
% non GUI windows.
%
% It allows the systems GUI window manager to place the window at its own
% discretion, that is, the (x,y) position of the windows top-left corner,
% as provided in the 'rect' parameter, is ignored and placement of the
% window is left to where the operating system deems it most ergonomical.
%

% This is the numeric constant for window placement by window manager:
rc = 2^19;

return;
