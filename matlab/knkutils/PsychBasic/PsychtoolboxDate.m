function d=PsychtoolboxDate
% PsychtoolboxDate returns the release date, e.g. '1 August 1998'.
% You can supply this string as an argument to DATNUM.

% HISTORY
% 12/3/04   awi    Copied into OS X  PTB. Added history comments.  

global Psychtoolbox
PsychtoolboxVersion; % Load the Psychtoolbox struct.
d=Psychtoolbox.date;
