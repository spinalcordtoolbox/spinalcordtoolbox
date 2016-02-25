function RestoreCluts
% RestoreCluts
%
% Restores the original CLUT's (aka gamma tables) in all graphics cards
% from backup copies made during calls to LoadIdentityClut() or
% BackupCluts(). This is mostly meant to undo changes made when operating
% high precision display devices and similar equipment.
%
% This is a no-operation if there aren't any CLUT's to restore.
%

% History:
% 05/31/08   mk  Written.

global ptb_original_gfx_cluts;

if isempty(ptb_original_gfx_cluts)
    % No backups available -- Nothing to do:
    return;
end

% Iterate over all backup cluts and reload them:
for screenid=0:length(ptb_original_gfx_cluts)-1
    if ~isempty(ptb_original_gfx_cluts{screenid + 1})
        % Restore clut for 'screenid':
        oldClut = ptb_original_gfx_cluts{screenid + 1};
        Screen('LoadNormalizedGammaTable', screenid, oldClut);
        ptb_original_gfx_cluts{screenid + 1} = [];
    end
end

% All cluts restored from backup: Release array with backups itself:
ptb_original_gfx_cluts = [];

% Done.
return;
