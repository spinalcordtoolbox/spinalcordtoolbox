function timebaseFrequencyHz = MachAbsoluteTimeClockFrequency

% timebaseFrequencyHz = MachAbsoluteTimeClockFrequency
%
% OS X: ___________________________________________________________________
%
% Return the frequency of the Mach Kernel "absolute timebase clock".  The
% frequency depends your  hardware, both the model of CPU and a system
% hardware clock.
% 
% Mach Kernel functions which assign real-time "Time constraint priority"
% status to threads give parameters in Mach time base units. The counter which
% clocks time allocated to your thread counts time in these units.  Use the
% absolute timebase clock frequency returned by MachAbsoluteTimeClockFrequency to convert
% seconds into absolute timebase units which you pass to functions which
% set which set priority:
% 
%   time_interval_in_mach_units= 
%        time_interval_in_seconds * timebaseFrequencyHz;
%
% For more information on the Mach absolute time clock see Apple Technical
% Q&A 1398:
%
%  http://developer.apple.com/qa/qa2004/qa1398.html
%
% OS 9: ___________________________________________________________________
%
% MachAbsoluteTimeClockFrequency is not provided on OS 9 because the Mach
% time base is a feature of only the OS X Mach Kernel.  
%
%
% WINDOWS: ________________________________________________________________
% 
% MachAbsoluteTimeClockFrequency is not provided on Windows because the
% Mach time base is a feature of only the OS X Mach Kernel.   
% _________________________________________________________________________
%
% see also: Priority

% HISTORY
% 4/6/05  awi     Wrote it. Based on Mario's changes to GetBusFrequencyMex
% 4/8/05  awi     Added link to Q&A 1398
% 4/8/05  awi     Updated "MachTimebase" to new name "MachAbsoluteTimeClockFrequency"

% On OS X MATLAb should find and execute a mex file of the same name, 
% not this help file.  So issue and error about a missing mex file if this 
% help file executes on OSX.
AssertMex('OSX');

% On every platform except OS X we arrive here. 
error('The Mach absolute time clock is a feature of only the Mac OS X Mach Kernel.');
