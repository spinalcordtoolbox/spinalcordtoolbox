function f = checkpathconflicts

% function f = checkpathconflicts
% 
% check for .m file conflicts in all directories found in the
% current path, excluding ones that are under matlabroot.
% report results to stdout.  return number of conflicts found.

% get all dirs
pathdirs = strsplit(path,pathsep);

% do it
f = 0;
for p=1:length(pathdirs)
  if isempty(strmatch(matlabroot,pathdirs{p}))
    f = f + checkpathconflicts_helper(pathdirs{p});
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function f = checkpathconflicts_helper(d)

% function f = checkpathconflicts_helper(d)
%
% check all *.m files found within directory <d> (no recursion).
% we are looking for duplicate .m files as determined via "which -all".
% if we find conflicts, report to stdout.  return the number of conflicts detected.
%
% note: we ignore "Contents.m", "demos.m", "Readme.m", and files beginning with ".".
%
% note: when calling which, we ignore entries starting with "/private/".

% report
fprintf(1,['checking: ',d,'\n']);

% do it
f = 0;
files = dir(fullfile(d,'*.m'));
for p=1:length(files)

  % get file name and remove .m suffix
  filename = files(p).name;
  filename = filename(1:end-2);

  % some acceptable exceptions:
  if isequal(filename,'Contents') | isequal(filename,'demos') | isequal(filename,'Readme') | ~isempty(strmatch('.',filename))
    continue;
  end

  % call "which"
  filewhich = which(filename,'-all');
  if isempty(filewhich)
    fprintf(1,[' weird, could not "which" on ',filename,'\n']);
  else

    % go through and filter the results
    idx = 1;
    while idx <= size(filewhich,1)
      if ~isempty(findstr('/private/',filewhich{idx}))    %| isequal('built-in',filewhich{idx}) | ~isempty(findstr('.mex',filewhich{idx}))
        filewhich(idx) = [];
      else
        idx = idx + 1;
      end
    end

    % finally, do the check
    if size(filewhich,1) > 1
      fprintf(1,[' function conflict detected for ',filename,'\n']);
      f = f + 1;
    end

  end

end
