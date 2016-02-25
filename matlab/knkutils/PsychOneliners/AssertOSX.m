function AssertOSX
% AssertOSX
% 
% OS X: ___________________________________________________________________
%
% Call AssertOSX at the head of scripts which use functions unique to the
% OS X Psychtoolbox.  AssertOSX will exit with an error if called from
% MATLAB running on any OS other than OS X, providing an explanation to
% users for why the script failed.  AssertOSX also serves to document your
% code clearly as being specific to OS X.    
%
% Use of OSX-specific Psychtoolbox functions is discouraged, though
% sometimes necessary.  
%
% Note that Psychtoolbox functions which are unique to OpenGL are a
% different category than those unique to OS X.  Please use AssertOpenGL in
% place of AssertOSX if your script calls functions specific to OpenGL but
% none specific to OS X. OpenGL-specific Psychtoolbox functions are any
% of the following:
%       Screen('MakeTexture'); 
%       Screen('DrawTexture');
%       Screen('Flip');
%
% Some Psychtoolbox functions are unique to OS X because they exploit
% features in OS X not present in other operating systems, for example,
% the MachPriority* functions underlying Priority in OS X. Generally, as
% with Priority, there is a platform-neutral function overlalying the
% platform-specific functions.  Where available, you can use these to write
% platform-neutral scripts, unless for your purposes you must take
% advantage of features specific to the OS.  PsychHID is another such
% example.  It exists only for OS X but is overlayed by KbCheck and Gamepad
% (a.k.a. Joystick) which also exist on OS 9.  If you have an unusual USB
% HID device, you can read from it in OSX using PsychHID, though compromising
% platform independence.  
%
% Overview- New functions for OS X fall into one of these categories:
%  * OpenGL-specific functions which are a permanent departure from earlier
%   Psychtoolboxes.  Call IsOpenGL or AssertOpenGL if you use only these. 
%  * Functions which take advantage of features unique to a specific
%  operating system.  When possible, avoid these by using platform-neutral
%   overlay functions. 
%  * Remaining differences are a temporary failure to keep the OS 9,
%  Windows, and OS X Psychtoolboxes synchronized during the course of
%  development. The AssertOSX script is itself an example of this; it has
%  not been ported back to OS 9 and Windows Psychtoolboxes yet.
%  Psychtoolbox help displays clear divisions denoting
%  platform-specificity.  In the case of not-yet-backported functions, these
%  divisions denote the actual currrent state of the Psychtoolbox, not its
%  intended design.  
%
%  You can make your scripts and functions platform-neutral by testing the
%  OS version using MATLAB's "computer" command and conditionally executing
%  platform-specific calls. 
%
% OS9: ___________________________________________________________________
%
% AssertOSX does not yet exist in OS 9. 
%
% WIN: ________________________________________________________________
% 
% AssertOSX does not yet exist in Windows.
% 
% _________________________________________________________________________
%
% see also: AssertOpenGL, IsOSX, computer

% HISTORY
%
% 7/15/04   awi Wrote it.
% 10/6/05 	awi Note here cosmetic changes by dgp between 7/15/04 and 10/6/05.

if ~IsOSX
    error('Use of an m file marked for execution only on OS X.  See ''help AssertOSX'' for more information');
end
