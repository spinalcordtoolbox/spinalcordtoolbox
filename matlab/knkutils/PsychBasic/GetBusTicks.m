function GetBusTicks
% ticks = GetBusTicks
% 
% OS X: ___________________________________________________________________
%
% Return the number of system bus ticks since startup.  The period of the
% bus tick clock depends on the model and frequency of your CPU. Use
% GetSecs instead for reliable, high-precision time measurement in standard
% units.
%
% Bus ticks returned by GetBusTicks are not the same as ticks returned by
% GetTicks.  The GetBusTicks clock is fast and precise, derived directly
% from the CPU clock and typically faster than 1/10 of its speed. On
% Allen's 1GHz G4 the GetBusTicks clock is 133 MHz.
%
% The GetTicks clock is slow, always 1/60.15 seconds. On OS
% X GetTicks is obsolete and provided only for compatability with older
% scripts.     
% 
% TIMING ADVICE: the first time you access any MEX function or M file,
% Matlab takes several hundred milliseconds to load it from disk.
% Allocating a variable takes time too. Usually you'll want to omit those
% delays from your timing measurements by making sure all the functions you
% use are loaded and that all the variables you use are allocated, before
% you start timing. MEX files stay loaded until you flush the MEX files
% (e.g. by changing directory or calling CLEAR MEX). M files and variables
% stay in memory until you clear them.
%
% OS 9: ___________________________________________________________________
%
% GetBusTicks does not exist in OS 9. 
%
% WINDOWS: ________________________________________________________________
% 
% GetBusTicks does not exist in Windows.
% 
% _________________________________________________________________________
%
% See also: GetBusTicksTick, GetSecs, GetSecsTick,  GetTicks, GetTicksTick.

% History
% 10/4/05		awi	began history.  

% There should be a mex file to match this .m documentation file.
AssertMex;
