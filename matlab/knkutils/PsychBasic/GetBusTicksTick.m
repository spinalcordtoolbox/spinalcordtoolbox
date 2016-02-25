function busTickPeriod = GetBusTicksTick

% busTickPeriod = GetBusTicksTick
%
% OS X: ___________________________________________________________________
%
% Return the period of the GetBusTicks clock.  The period of the
% GetBusTicks clock depends on your model of CPU and its clock speed.
% For reliable high-precision timing in standard units of seconds use
% GetSecs instead of GetBusTicks.
%
% GetBusTicksTick returns the period of the GetBusTicks clock.  The
% frequency is found in the hw.busfreq field of the struct returned by
% Screen('Computer').
%
% OS 9: ___________________________________________________________________
%
% GetBusTicksTick does not exist in OS 9. 
%
% WINDOWS: ________________________________________________________________
% 
% GetBusTicksTick does not exist in Windows.
% 
% _________________________________________________________________________
%
% SEE ALSO: GetBusTicks, GetSecs, GetSecsTick, Screen('Computer') 

%   HISTORY:
%   04/18/03    awi    Wrote GetBusTick.m
%   7/10/04     awi    Improved.        
%   7/12/04     awi    Added OS 9 and Windows sections.

if IsOSX
    cInfo=Screen('Computer');
    busTickPeriod=1/cInfo.hw.busfreq;
else
    error(['GetBusTick function not implemented on platform ' computer ]);
end

