function fName=NameFrequency(fValue, numDecimalPlaces)
% fName=NameFrequency(fValue, [numDecimalPlaces])
%
% NameFrequency accepts a frequency (in Hz) and returns a
% string naming that frequency in a more readable form.  For example:
%
%     >> c=Screen('Computer');
%     >> c.hw.cpufreq
% 
%     ans =
% 
%        999999997
% 
%     >> NameFrequency(c.hw.cpufreq)
% 
%     ans =
% 
%     1.00 GHz
% 
%     >> NameFrequency(c.hw.cpufreq,6)
% 
%     ans =
% 
%     999.999997 MHz
% 
%     >>
%
%
% By default NameFrequency displays two digits to the right of the decimal
% point. Specify other numbers of digits by passing the optional
% numDecimalPlaces argument.  
%
% NameFrequency is used by DescribeComputer to name the clock frequency of
% the CPU. For that purpose, it displays frequency to a specified number of
% decimal places, not to a specified precision.  It is therfore a poor
% choice for scientific work, where typcially a fixed precision in digits
% is appropriate.  
%
% see also: NameBytes, DescribeComputer

% HISTORY 
% 12/17/04   awi  Wrote it.

 
if nargin < 2
    numDecimalPlaces=2;
end

fNames={
'Hz',
'KHz',
'MHz',
'GHz',
'THz',
'PHz',
'EHz',
'ZHz',
'YHz'
};

numNames=length(fNames);
foundIndex=0;
for i=1:numNames
    if round(fValue/10^((i-1)*3) * (10^numDecimalPlaces)) / (10^numDecimalPlaces) < 1000.00
        namesIndex=i;
        foundIndex=1;
        break;
    end
end
if ~foundIndex
    error('Failed to find a name for the value');
end
        
unitsName=fNames{namesIndex};
sprintFormat=['%3.' int2str(numDecimalPlaces) 'f'];
fName=[sprintf(sprintFormat, fValue/10^((namesIndex-1)*3)) ' ' unitsName];  

