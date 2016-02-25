function number = GetNumber(varargin)
% number = GetNumber([deviceIndex][, untilTime=inf][, optional KbCheck arguments...])
% 
% Get a number typed at the keyboard. Entry is terminated by
% <return> or <enter>. Typed keys are not echoed. Useful for
% i/o in a Screen window. Equivalent to "number=str2num(GetString)".
%
% Returns the empty matrix if no number is entered. Returns a
% column vector with multiple numbers if more than one number
% is entered.
%
% See also: GetEchoNumber, GetString, GetEchoString

% 12/7/95	dhb	Wrote it in response to query from Tina Beard.
% 3/15/97	dgp	Replaced sscanf by str2num, which copes better with nonnumeric
%				      input, returning an empty matrix instead of a null string.
% 3/15/97	dgp	Call GetString instead of doing the work here.
% 3/17/97   dhb Got rid of obsolete 's' interface.
% 10/22/10  mk  Switch to use of KbGetChar for keyboard input.

number = str2num(GetString(1, varargin{:})); %#ok<ST2NM>
