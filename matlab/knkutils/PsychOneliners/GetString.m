function string = GetString(useKbCheck, varargin)
% string = GetString([useKbCheck=0][, deviceIndex][, untilTime=inf][, optional KbCheck arguments...])
% 
% Get a string typed at the keyboard. Entry is terminated by 
% <return> or <enter>.
%
% If the optional flag 'useKbCheck' is set to 1 then KbCheck is used - with
% potential optional additional 'KbCheck args...' for getting the string
% from the keyboard. Otherwise GetChar is used. 'useKbCheck' == 1 is
% restricted to standard alpha-numeric keys (characters, letters and a few
% special symbols). It can't handle all possible characters and doesn't
% work with non-US keyboard mappings. Its advantage is that it works
% reliably on configurations where GetChar may fail, e.g., on MS-Vista and
% Windows-7.
%
% Useful for i/o in a Screen window. Typed keys are not echoed.
%
% See also: GetEchoString, GetNumber, GetEchoNumber
%

% 12/7/95 dhb	Wrote GetNumber in response to query from Tina Beard.
% 12/8/95 dhb	Add delete functionality.
% 2/4/97  dhb	Fixed bug.  Can now hit delete more than once.
% 2/5/97  dhb	Accept <enter> as well as <cr>.
%         dhb	Allow string return as well.  
% 3/15/97 dgp Created GetString based on dhb's GetNumber.
% 3/31/97 dhb Fix bug arising from new initialization.
% 2/28/98 dgp Use GetChar instead of obsolete GetKey. Use SWITCH and LENGTH.
% 3/27/98 dhb Fix bug from 2/28/98, put abs around char in switch.
% 12/19/06 mk Adapted for use with PTB-3.
% 10/22/10  mk        Optionally allow to use KbGetChar for keyboard input.

string = '';

if nargin < 1
    useKbCheck = [];
end

if isempty(useKbCheck)
    useKbCheck = 0;
end

if ~useKbCheck
    % Flush the keyboard buffer:
    FlushEvents;
end

while 1	% Loop until <return> or <enter>
    if useKbCheck
        char = GetKbChar(varargin{:});
    else
        char = GetChar;
    end
    
    if isempty(char)
        string = '';
        return;
    end
    
	switch(abs(char))
		case {13,3,10},	% <return> or <enter>
			break;
		case 8,			% <delete>
			if ~isempty(string)
				string=string(1:length(string)-1);
			end
		otherwise,
			string=[string char]; %#ok<AGROW>
	end
end
