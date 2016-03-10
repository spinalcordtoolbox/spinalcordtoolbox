function ret = RestrictKeysForKbCheck(enablekeys)
% Restrict operation of KbCheck et al. to a subset of keys on the keyboard.
%
% oldenablekeys = RestrictKeysForKbCheck([enablekeys])
%
% Specify a vector of keycodes for keys which should be
% checked/used by KbCheck and KbWait. This is useful to
% only enable specific keys, e.g., to save time. The function
% returns the old set of enabled keys.
%
% Example: To enable the keys with keycodes 4, 6 and 7, do
% RestrictKeysForKbCheck([4, 6, 7]);
%
% Calling RestrictKeysForKbCheck([]); ie., with an empty vector, will
% reenable all keys.
%
% Caution: This setting is reset to "empty" during a "clear all" command,
% ie., all keys will be enabled again after a "clear all"!
%
% Background info:
%
% Some users of Laptops experienced the problem of "stuck keys": Some keys
% are always reported as "down", so KbWait returns immediately and KbCheck
% always reports keyIsDown == 1. This is often due to special function keys.
% These keys or system functionality are assigned vendor specific
% key codes, e.g., the status of the Laptop lid (opened/closed) could be
% reported by some special keycode. Whenever the Laptop lid is open, this key
% will be reported as pressed. You can work around this problem by passing
% a subset of keycodes to be used by KbCheck and KbWait, whereas all other
% unwanted keys are ignored.
%
% Another advantage is a significant speed gain for KbCheck et al. on
% MacOS/X systems, where the execution time of KbChecks is proportional to
% the number of keys to check.
% _________________________________________________________________________
% 
% See also: FlushEvents, KbName, KbDemo, KbWait, KbCheck, GetChar, CharAvail.

% History:
% 11/16/08 Written (MK).

% This global variable is used to pass the vector of enabled keys
% to the KbCheck command:
global ptb_kbcheck_enabledKeys;

% Return old vector:
ret = find(ptb_kbcheck_enabledKeys > 0);

if nargin>0
   % Set new vector:
   if isempty(enablekeys)
       ptb_kbcheck_enabledKeys = [];
   else
       ptb_kbcheck_enabledKeys = zeros(1,256);
       ptb_kbcheck_enabledKeys(enablekeys) = 1;
   end
end

return;
