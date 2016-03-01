function ticks=GetTicks
% ticks = GetTicks
% 
% The number of system ticks since startup. One tick is 1/60.15 second.
% 
% For more precise timing use GetSecs or WaitSecs.
% 
% TIMING ADVICE: the first time you access any MEX function or M file,
% Matlab takes several hundred milliseconds to load it from disk.
% Allocating a variable takes time too. Usually you'll want to omit
% those delays from your timing measurements by making sure all the
% functions you use are loaded and that all the variables you use are
% allocated, before you start timing. MEX files stay loaded until you
% flush the MEX files (e.g. by changing directory or calling CLEAR
% MEX). M files and variables stay in memory until you clear them.
% 
% See also: WaitTicks, GetTicksTick, GetSecs, GetSecsTick, WaitSecs, GetChar, GetBusTicks, GetBusTicksTick.

% 5/7/96  dgp  Corrected the definition of a tick.
% 1/29/97 dhb  More comments.
% 3/15/97 dgp  Expanded comments.
% 7/12/04 awi  ****** OS X-specific fork from the OS 9 version *******
%               Fixed function definition at top, it omitted return
%               argument. Added GetBusTicks, GetBusTicksTick and
%               GetTicksTick to see also.

AssertMex('GetTicks.m');
