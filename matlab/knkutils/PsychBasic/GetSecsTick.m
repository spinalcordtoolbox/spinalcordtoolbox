function tickDuration=GetSecsTick
% tickDuration=GetSecsTick
% 
% GetSecsTick returns the fraction of a second which is a tick of the GetSecs clock. 
% 
% See also: GetSecs, WaitTicks, GetTicks, GetTicksTick, GetBusTicks, GetBusTicksTick.

% 3/15/02  awi  wrote it.
% 4/02/02  awi  Changed the return value to the clock period from the clock frequency. 
%               Changed the name from "GetTimeBase" to "GetSecsTick".
%               Added script provided by Denis for OS9 version of the Psychtoolbox
% 7/10/04  awi  Changed for OSX by adding GetBusTicks and GetBusTicksTick
%               to see also. Replaced streq with AssertMex and isOS9.
% 10/4/05  awi  Note here cosmetic changes by dgp between 7/10/04 and 10/4/05. 

% On Windows and OSX this is only a help file and should not execute.
AssertMex('Win','OSX');

if isOS9
	available=Screen('Preference','Available');
        if available.UpTime
           tickDuration=1e-9; % UpTime tick (a rough guess)
		else
           tickDuration=8e-6; % Microseconds tick (a rough guess)
	end   
else
    error('Platform unsupported by the Psychtoolbox');
end

