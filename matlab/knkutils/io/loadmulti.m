function f = loadmulti(m,var,dim,gentle)

% function f = loadmulti(m,var,dim,gentle)
%
% <m> is a pattern that matches one or more .mat files (see matchfiles.m)
% <var> is a variable name
% <dim> (optional) is
%   N means the dimension along which to concatenate
%   0 means do not concatenate; instead, successively merge
%     in new versions of <var>.  there are two cases:
%     if <var> is a regular matrix, NaN elements get 
%     overwritten with new content.  if <var> is a cell 
%     matrix, elements that are [] get overwritten with
%     new content.  note that the dimensions
%     of <var> should be the same in each file.
%     however, if they aren't, that's okay --- we 
%     automatically expand to the bottom and the right
%     as necessary.
%   default: 0.
% <gentle> (optional) is
%   0 means die if <var> is not found in a given file.
%   1 means continue on (and do not crash) if <var> is
%     not found in a given file.
%   default: 0.
%
% get <var> from the file(s) named by <m> and either
% concatenate different versions together or merge 
% different versions together.
%
% we issue a warning if no files are named by <m>.
%
% we report progress to the command window if the load
% takes longer than 10 seconds.
%
% example:
% a = {1 2 []};
% save('atest1.mat','a');
% a = {[] 5 3};
% save('atest2.mat','a');
% isequal(loadmulti('atest*mat','a',0),{1 2 3})

% internal constants
toolong = 10;  % seconds

% input
if ~exist('dim','var') || isempty(dim)
  dim = 0;
end
if ~exist('gentle','var') || isempty(gentle)
  gentle = 0;
end

% transform
m = matchfiles(m);

% check sanity
if length(m)==0
  warning('no file matches');
  f = [];
  return;
end

% do it
stime = clock; didinit = 0; isfirst = 1;
for p=1:length(m)

  % report status
  if etime(clock,stime) > toolong
    if didinit
      statusdots(p,length(m));
    else
      didinit = 1;
      fprintf('loadmulti');
    end
  end

  % get values from the file
  loaded = load(m{p},var);
  varnames = fieldnames(loaded);
  if isempty(varnames)
    if gentle==0
      error(sprintf('loadmulti: <var> was not found in some particular file (%s)',m{p}));
    end
  else
    vals = getfield(loaded,varnames{1});
  
    % special case
    if dim==0
      
      if isfirst
        f = vals;
        isfirst = 0;
      else
        [f,vals] = equalizematrixdimensions(f,vals);
        if iscell(f)
          ix = cellfun(@isempty,f);
        else
          ix = isnan(f);
        end
        f(ix) = vals(ix);
      end
  
    % usual case
    else
      if isfirst
        f = vals;
        isfirst = 0;
      else
        f = cat(dim,f,vals);
      end
    end
  end

end

% report
if etime(clock,stime) > toolong
  fprintf('done.\n');
end
