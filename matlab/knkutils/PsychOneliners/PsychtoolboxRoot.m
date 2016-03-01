function path=PsychtoolboxRoot
% path=PsychtoolboxRoot
% Returns the path to the Psychtoolbox folder, even if it's been renamed.
% Also see matlaboot, DiskRoot, [and maybe DesktopFolder].

% 6/29/02 dgp Wrote it, based on a suggestion by David Jones <djones@ece.mcmaster.ca>.
% 9/10/02 dgp Cosmetic.
% 1/24/08 mpr modified help because MatlabRoot doesn't work, DesktopFolder
%               appears no longer to exist, and neither did DiskRoot, but I'd 
%               already written a function to do what that did so I donated it 

path=which('PsychtoolboxRoot');
i=find(filesep==path);
path=path(1:i(end-1));
