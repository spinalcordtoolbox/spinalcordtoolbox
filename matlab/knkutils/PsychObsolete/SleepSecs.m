% SleepSecs(s)
%  
% Wait for duration s seconds, up to one second.  SleepSecs suspends the
% MATLAB process.     
%   
% OS X: ___________________________________________________________________
%
% If you set the priority level to greater than 0 using either Priority or
% Rush then use SleepSecs instead of WaitSecs while priority is elevated.
% Whereas SleepSecs surrenders CPU time,  WaitSecs consumes CPU time,
% exceeding limits set by Priority or Rush and causing the Mach kernel to
% revoke any priority setting greater than 0.
% 
% If you are playing an animation, then use Screen('Flip') to both
% synchronize updating of the display to the Video BLanking invterval (VBL)
% and to delay your animation loop until the next VBL; Like SleepSecs, Flip
% surrenders CPU time to other processes, abiding by limits set when
% negotiating with the kernel for priority levels > 0.
% 
% You should not need to continuously sleep the MATLAB process at high
% priority for periods greater than 1 second.  If you feel the need for 
% continuous  delay at elevated priority for greater than the maximum one
% second duration of SleepSecs, consider instead lowering priority to 0,
% calling WaitSecs, and then reeleveting priority.
% 
% SleepSecs would be useful in a loop which called KbCheck at high priority
% while not displaying an animation. 
%
% SleepSecs uses the Posix usleep ("microsleep") function.
%
% OS 9: ___________________________________________________________________
%
% SleepSecs does not exist in OS 9. 
%
% WINDOWS: ________________________________________________________________
% 
% SleepSecs does not exist in Windows.
% 
% _________________________________________________________________________
%
% See Also: Priority, Rush, SetMachPriorityMex, GetMachPriorityMex, Screen('Flip')
%

% 7/10/04   awi     Wrote it.
% 7/12/04   awi     Added sections.   
%

error('SleepSecs is obsolete.  Use WaitSecs instead.');

