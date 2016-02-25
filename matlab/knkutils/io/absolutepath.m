function f = absolutepath(x)

% function f = absolutepath(x)
%
% <x> is a string referring to a file.  can be absolute or
%   relative to the current directory.
%
% return the absolute path to the file.
% we rely on pwd to figure out the path.
% note: the file does not actually have to exist.
%
% example:
% absolutepath('temp')

% find any /
locs = strfind(x,filesep);

% if none, be relative to current directory and get out
if isempty(locs)
  f = [pwd filesep x];
  return;
end

% save old cd
oldcd = cd;

% change to new
cd(x(1:locs(end)));

% what is the filename
remain = x(locs(end)+1:end);

% return
cwd = pwd;
if isequal(cwd,filesep)
  f = [cwd remain];
else
  f = [cwd filesep remain];
end

% change back to old cd
cd(oldcd);
