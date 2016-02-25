% Method to override a MATLAB built-in function with a user-supplied function 
% with the same name. The way this works is that it replaces the built-in function
% with a function handle parameter whose name matches that of the overriden
% function.
%
% Usage:
%       functionName = overrideBuiltInFunction('functionName', overridingFunctionPath);
%
%       where the overridingFunctionPath string can be the full path to the overriding
%       function or a uniquely-identifying subset of the full path.
%
%       This call must be done in the script where you will use the override. 
%       To use across several scripts, make it a global function handle:
%
%       global functionName
%       functionName = overrideBuiltInFunction('functionName', overridingFunctionPath);
%
% Example Usage: 
%       Override Matlab's built-in function lab2xyz with the one supplied by ISETBIO
%       located in '/Users/Shared/Matlab/Toolboxes/ISETBIO/isettools/color/transforms'
%
%       lab2xyz = overrideBuiltInFunction('lab2xyz', 'isetbio');
%
% Test that it works:
%       clear all
%       functions(@lab2xyz)
%       lab2xyz = overrideBuiltInFunction('lab2xyz', 'isetbio');
%       functions(lab2xyz)
%
% Undoing the override:
%       To 'unoverride', simply clear the function handle. This should bring back
%       the built-in function, e.g.:
%
%       clear lab2xyz
%
%
% 10/9/2014   NPC  Wrote it.
%

function functionHandle = overrideBuiltInFunction(functionName, userPath)
    [paths, status] = which(functionName, '-all');
    
    if isempty(paths)
        error('''%s'' was not overriden. Function does not exist anywhere.\n', functionName); 
    end
    
    k = 0;
    foundInUserPath = false;
    while ((k < numel(paths)) && (~foundInUserPath))
        k = k + 1;
        if ~isempty(strfind(paths{k}, userPath))
            foundInUserPath = true;
        end
    end
    
    if (foundInUserPath)
        fullFunctionName = char(paths{k});
        [pathstr,name,~] = fileparts(fullFunctionName);
        localDir = pwd;
        cd(pathstr);
        functionHandle = str2func(name);
        cd(localDir);
    else
       error('''%s'' was not overriden. Not found in the specified path (''%s'').\n', functionName, userPath); 
    end
end
