function s=GetSecs(s,subscript)
% s=GetSecs
% 
% GetSecs returns the time in seconds (with high precision). GetSecs uses
% the highest precision realtime clock on each operating system. The zero
% point (point in time where GetSecs would report a value of zero) is
% operating system dependent and nothing to be relied on.
%
% 
% TIMING ADVICE: The first time you access any MEX function or M file,
% Matlab takes several hundred milliseconds to load it from disk.
% Allocating a variable takes time too. Usually you'll want to omit
% those delays from your timing measurements by making sure all the
% functions you use are loaded and that all the variables you use are
% allocated, before you start timing. MEX files stay loaded until you
% flush the MEX files (e.g. by changing directory or calling CLEAR
% MEX). M files and variables stay in memory until you clear them.
% 
% Win : ___________________________________________________________________
%
% On Windows machines the high precision QueryPerformanceCounter() call 
% is used to get the number of seconds since system start up, if a 
% performance counter is available. Otherwise, or if the high precision timer
% is found to be defective or unreliable, the less accurate timeGetTime()
% system call is used. Some windows systems and pc's are known to have
% defective or unreliable timing facilities under some conditions.
% Psychtoolbox tries to detect and handle such systems at runtime - it
% performs runtime consistency checks. For a more thorough test, run
% GetSecsTest. See also the FAQ section of the Psychtoolbox Wiki for more
% background info, as well as "help GetSecsTest". Resolution of time on
% Windows varies: If the high precision clock is used, it will be
% microsecond resolution and accuracy, if the fallback clock is used, it
% will be roughly millisecond resolution and accuracy.
% 
% OSX : ___________________________________________________________________
%
% On machines running Apples OSX, the mach_absolutetime() call is used,
% which provides at least microsecond accuracy and resolution. To our
% current knowledge, all Macintosh computers have reliably working clocks.
%
% LINUX : _________________________________________________________________
%
% On Linux, the gettimeofday() system call is used, which usually has
% microsecond resolution and accuracy. Linux always chooses the highest
% precision clock on a system for that call, usually the processors
% performance counter or the HPET high precision event timer, or the ACPI
% power management timer - whatever is the best tradeoff between
% reliability, acccuracy and performance. To our current knowledge, all
% computers running a Linux 2.6 kernel have reliably working clocks.
%
%
% See also: WaitSecs, GetSecsTest, 

% 3/15/97  dgp  Expanded comments.
% 4/9/97   dgp  Documented the optional arguments.
% 2/17/99  dgp  Typo: replaced s[subscript] by s(subscript).
% 3/15/99  dgp  Mention SecondsMultiplier.
% 3/15/99  xmz  Put in comments for Windows version.
% 3/23/99  dgp  Cosmetic.
% 4/10/99  dgp  Say more about Mac OS Time Manager.
% 4/19/99  dgp  Begin with an executive summary, as suggested by Allen Ingling.
% 2/4/00   dgp  Updated for Mac OS 9.
% 6/16/00  dgp  Cosmetic.
% 10/25/05 awi  Divided into general section and OS 9 & Win specific sections.
%               Imported into OS X PTB.
% 01/28/08 mk   Updated help texts to match current implementation.

AssertMex('GetSecs.m');
