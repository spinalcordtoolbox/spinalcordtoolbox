function AddToMatlabPathDynamically(directory)
% AddToMatlabPathDynamically(directory)
%
% Add the directory and its subdirectories to Matlab's path, dynamically,
% with strippint out of .svn and .git directories.
% 
% Use for for putting routines onto path that are specifict to a particular
% project, without them staying around and clogging up the name space.
%
% Typical usages:
% a) When getting version info
%   exp.mFileName = mfilename;
%   [exp.versionInfo,exp.codeDir] = GetAllVersionInfo(exp.mFileName);
%   AddToMatlabPathDynamically(exp.codeDir);
%
% b) Direct call
%   AddToMatlabPathDynamically( fileparts(which(mfilename))); 
%
% 7/12/13  dhb  Wrote it.
% 7/25/14  dhb  Make independent of BrainardLab idiosyncracies.

%% Dynamically add the program code to the path if it isn't already on it.
if isempty(strfind(path, directory))
	fprintf('- Adding %s dynamically to the path...', directory);
    thePath = RemoveMatchingPaths(genpath(directory),[filesep '.svn']);
    thePath = RemoveMatchingPaths(thePath,[filesep '.git']);
	addpath(thePath, '-end');
	fprintf('Done\n');
end
