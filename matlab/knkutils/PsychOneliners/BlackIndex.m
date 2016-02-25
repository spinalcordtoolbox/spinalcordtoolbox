function black=BlackIndex(windowPtrOrScreenNumber)
% color=BlackIndex(windowPtrOrScreenNumber)
% Returns the CLUT index to produce black at the current screen depth,
% assuming a standard color lookup table for that depth. E.g.
% black=BlackIndex(w);
% Screen(w,'FillRect',black);
% 
% See WhiteIndex.
% 
% When the screen is 1 to 8 bit mode, the Macintosh OS always makes the
% first clut element white and the last black. In 16 or 32 bit mode the
% clut goes from black to white. These CLUT conventions can be overridden
% by Screen 'SetClut', which makes a direct call to the video driver,
% bypassing the Mac OS, allowing you to impose any CLUT whatsoever.

% 3/10/98	dgp Wrote it.
% 3/30/98	dgp Consider only one channel, even for 16 and 32 bit modes.
% 3/8/2000	emw Added platform conditional
% 3/8/2000	dgp Fixed platform conditional
% 3/30/2004 awi Added OS X case. For now OS X only supports true-color mode, so
%               BlackIndex behavior on OS X will have to change when we add
%               more depth modes.
% 1/29/05   dgp Cosmetic.
% 3/01/08   mk  Changed to a simple return value of zero - anything else
%               doesnt' make sense with PTB-3.

if nargin~=1
	error('Usage: color=BlackIndex(windowPtrOrScreenNumber)');
end

black=0;

return;
