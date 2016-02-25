function theFolder=FunctionFolder(functionName)

% filePath=FunctionFolder(functionName)
%
% Accepts the name of a function and returns the fullpath to its enclosing
% folder.  The path is the same as that returned by MATLAB's which command,
% minus the name of the function itself.

% HISTORY
% 1/27/05   awi     Wrote it for a use by mex files.  MATLAB callbacks
%                   require several lines in C, so even though FunctionFolder does not do a
%                   lot, saves some time.  Used by StoreBit and maybe in the future GetChar.


pathToFile=which(functionName);
theFolder=fileparts(pathToFile);



