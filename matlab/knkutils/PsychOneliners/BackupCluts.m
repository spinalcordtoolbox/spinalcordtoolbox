function BackupCluts(screenIds)
% Backup current clut gamma tables of screens for later restoration via RestoreCluts().
%
% Usage:
%
% BackupCluts([screenIds=all]);
%
% Saves all cluts/gamma tables of all connected display screens by default,
% or of specified vector 'screenIds' to a global variable. The cluts can be
% restored from this backup copy by simply calling "RestoreCluts". This is
% also done automatically if "sca" is executed at the end of a session.
%

% History:
% 28.09.2010   mk   Written.

% Store backup copies of clut's for later restoration by RestoreCluts():
global ptb_original_gfx_cluts;

% Create global clut backup cell array if it does not exist yet:
if isempty(ptb_original_gfx_cluts)
    % Create 10 slots for out up to 10 screens:
    ptb_original_gfx_cluts = cell(10,1);
end

% If no specific vector of screenids given, backup all screens:
if nargin < 1
    screenIds = [];
end

if isempty(screenIds)
    screenIds = Screen('Screens');
end

for screenid = screenIds
    % Do we have already a backed up original clut for 'screenid'?
    % If so, we don't store this clut, as an earlier invocation of a clut
    % manipulation command will have stored the really real original lut:
    if isempty(ptb_original_gfx_cluts{screenid + 1})
        % Nope:

        % Retrieve current clut for screen 'screenid':
        oldClut = Screen('ReadNormalizedGammaTable', screenid);

        % Store backup:
        ptb_original_gfx_cluts{screenid + 1} = oldClut;
    end
end

return;
