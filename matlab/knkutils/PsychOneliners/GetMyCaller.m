function caller=GetMyCaller
% caller=GetMyCaller
%
% GetMyCaller returns the name of its caller by using MATLAB's dbstack
% function.  If invoked from the command line, then GetMyCaller returns
% 'base'.  
%
% see also: dbstack
%
% 2004     awi Wrote it.
% 1/29/05  dgp Cosmetic.
% 10/24/05 awi Cosmetic.
% 01/13/07 dhb Update comments for PTB-3 universe.

[callStack, stackIndex]=dbstack;
callDepth=length(callStack);
if callDepth==1
    caller='base';
else
    fullCaller=callStack(2).name;
    [pathstr,caller,extension,version]=fileparts(fullCaller);
end

