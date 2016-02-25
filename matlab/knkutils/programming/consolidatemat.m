function results = consolidatemat(files,outfile,varsexclude)

% function results = consolidatemat(files,outfile,varsexclude)
%
% <files> is a wildcard matching one or more .mat files
% <outfile> (optional) is a .mat file to write 'results' to.
%   if [] or not supplied, don't write to a .mat file.
% <varsexclude> (optional) is a variable name or a cell vector of
%   variable names to NOT load.  if [] or not supplied, load everything.
%
% we use matchfiles.m to match the <files>.
% we then construct a struct array with elements 
%   containing the results of loading each .mat file.
% this array is named 'results' and we save it
%   to <outfile> if supplied.
%
% example:
% a = 1; b = 2; save('test001.mat','a','b');
% a = 3; b = 4; save('test002.mat','a','b');
% consolidatemat('test*.mat','final.mat');
% results = loadmulti('final.mat','results');
% results(1)
% results(2)

% TODO: what about mismatches in the contents of the files?
%       save only the intersection?  report to screen?

% input
if ~exist('outfile','var') || isempty(outfile)
  outfile = [];
end
if ~exist('varsexclude','var') || isempty(varsexclude)
  varsexclude = [];
end

% do it
files = matchfiles(files);
clear results;
fprintf('consolidatemat: ');
for p=1:length(files)
  statusdots(p,length(files));
  if isempty(varsexclude)
    a = load(files{p});
  else
    a = loadexcept(files{p},varsexclude,1);
  end
  if exist('results','var')
    assert(isequal(sort(fieldnames(results(1))),sort(fieldnames(a))), ...
           sprintf('unexpected fields in file "%s"',files{p}));
  end
  results(p) = a;
end
if ~isempty(outfile)
  fprintf('saving...');
  save(outfile,'results');
end
fprintf('done.\n');
