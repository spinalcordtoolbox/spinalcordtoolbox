function tickPeriod = GetTicksTick

%  tickPeriod = GetTicksTick
%
%  Return the period of the GetTicks clock. The period is always 1/60.15
%  seconds by definition.
%
% SEE ALSO: GetTicks, GetSecs, GetSecsTick, GetBusTicks, GetBusTicksTick. 

%   HISTORY:
%   07/10/04    awi     Wrote GetBusTick.m
%   07/12/04    awi     Added platform sections.
%   10/4/05		awi     Noted here cosmetic changes by dgp between 07/12/04 and 10/4/05

tickPeriod=1/60.15;
