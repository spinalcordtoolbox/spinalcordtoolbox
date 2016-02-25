function pid=GetPID

% pid=GetPID
%
% Return the process ID of the MATLAB process.  GetPID calls the POSIX
% function "getpid".  
%

PsychAssertMex;
