function f = strsplit(str,pattern)

% function f = strsplit(str,pattern)
%
% <str> is a string
% <pattern> (optional) is a string.  default: sprintf('\n').
% 
% split <str> using <pattern>.  return a cell vector of string fragments.
% note that we generate beginning and ending fragments.
%
% example:
% isequal(strsplit('test','e'),{'t' 'st'})
% isequal(strsplit('test','t'),{'' 'es' ''})

% input
if ~exist('pattern','var') || isempty(pattern)
  pattern = sprintf('\n');
end

% find indices of matches
indices = strfind(str,pattern);

% do it
cnt = 1;
f = {};
for p=1:length(indices)
  temp = str(cnt:indices(p)-1);
  f = [f {choose(isempty(temp),'',temp)}];
  cnt = indices(p)+length(pattern);
end
temp = str(cnt:end);
f = [f {choose(isempty(temp),'',temp)}];
