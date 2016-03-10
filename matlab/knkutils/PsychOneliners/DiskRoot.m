function ThisDisk = DiskRoot
% DiskName = DiskRoot
%
% Purpose:  Find the name of the disk for the current working directory.

% History:
% 1/24/08		mpr scavenged the code from his personal files

CurDir = pwd;
SepPos = find(CurDir == filesep);
ThisDisk = CurDir(1:SepPos(1));

return;
