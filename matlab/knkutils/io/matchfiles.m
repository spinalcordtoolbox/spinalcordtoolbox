function f = matchfiles(patterns,sorttype)

% function f = matchfiles(patterns,sorttype)
%
% <patterns> is
%   (1) a string that matches zero or more files or directories (wildcards '*' okay)
%   (2) the empty matrix []
%   (3) a cell vector of zero or more things like (1) or (2)
% <sorttype> (optional) is how to sort in each individual match attempt.
%   't' means sort by time (newest first)
%   'tr' means sort by time (oldest first)
%   default is [], which means to sort alphabetically by explicitly using MATLAB's sort function.
%   (note that MATLAB's sort function may sort differently than UNIX's ls function does!)
%
% return a cell vector of strings containing paths to the matched files and/or directories.
% if there are no matches for an individual match attempt, we issue a warning.
%
% this function should be fully functional on Mac and Linux.  however, on Windows,
% we have the following limitations:
% - you cannot use the '?' operator
% - you can use the '*' operator only once and at the end of the expression
%   (not in an intermediate directory)
%
% on Mac and Linux, if we run into the too-many-files limitation of the ls command,
% we will resort to the alternative mode described above, and this inherits the 
% same limitations.
%
% history:
% 2011/09/28 - if ls returns too many files, resort to alternative.  also, the alternative mode now allows sorttype to be specified.
% 2011/08/07 - allow empty matrix as an input
% 2011/04/02 - now, works on Windows (in a limited way)
% 2011/04/02 - oops, time-sorting behavior did not work.  bad bug!!!
% 2011/02/24 - escape spaces in patterns using \.  this fixes buggy behavior.
% 2011/01/21 - explicitly use MATLAB's sort function to ensure consistency across platforms.

% input
if ~exist('sorttype','var') || isempty(sorttype)
  sorttype = '';
end
if ~iscell(patterns)
  patterns = {patterns};
end
% % if ~isunix
% %   assert(isempty(sorttype),'due to current implementation limitations, <sorttype> must be [] on Windows');
% % end

% do it
f = {};
for p=1:length(patterns)
  if isempty(patterns{p})
    continue;
  end
  
  % if UNIX, try to use ls
  doalternative = 0;
  if isunix
    [status,result] = unix(sprintf('/bin/ls -1d%s %s',sorttype,regexprep(patterns{p},' ','\\ ')));
    if status==126  % oops, too many files
      doalternative = 1;
    elseif status~=0
      warning(sprintf('failure in finding the files or directories for %s',patterns{p}));
    else
      temp = strsplit(result,sprintf('\n'));
      temp = temp(~cellfun(@(x) isempty(x),temp));  % remove empty entries
      if isempty(sorttype)
        temp = sort(temp);
      end
      f = [f temp];
    end
  end
  
  % if not UNIX or if we failed by matching too many files using ls, we have to do the alternative
  if ~isunix || doalternative
    if exist(patterns{p},'dir')
      f = [f {patterns{p}}];
    else
      tempdir = stripfile(patterns{p});
      dmatch = dir(patterns{p});
      if isempty(dmatch)
        warning(sprintf('failure in finding the files or directories for %s',patterns{p}));
      else
        if isequal(sorttype,'t')
          [ss,ii] = sort(cat(2,dmatch.datenum),2,'descend');
        elseif isequal(sorttype,'tr')
          [ss,ii] = sort(cat(2,dmatch.datenum));
        else
          [ss,ii] = sort(cat(2,{dmatch.name}));
        end
        dmatch = dmatch(ii);
        temp = cat(2,{dmatch.name});
        temp = temp(~cellfun(@(x) isempty(x),temp));  % remove empty entries
        temp = temp(~cellfun(@(x) isequal(x(1),'.'),temp));  % remove things starting with .
        if ~isempty(tempdir)
          temp = cellfun(@(x) [tempdir x],temp,'UniformOutput',0);  % add directory
        end
        f = [f temp];
      end
    end
  end
end
