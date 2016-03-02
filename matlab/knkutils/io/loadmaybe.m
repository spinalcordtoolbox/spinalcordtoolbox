function [f,success] = loadmaybe(m,var)

% function [f,success] = loadmaybe(m,var)
%
% <m> is a pattern that matches exactly one .mat file (see matchfiles.m)
% <var> is a variable name
%
% attempt to load <var> from <m>, suppressing warnings.
% if the variable exists, return <f> with the value and <success> as 1.
% if the variable doesn't exist, return <f> as [] and <success> as 0.
%
% example:
% a = 1;
% save('atest1.mat','a');
% [f,success] = loadmaybe('atest1.mat','blah');
% isequal(success,0)

% transform
m = matchfiles(m);

% check sanity
assert(length(m)==1,'<m> does not match exactly one file');

% get values from the file
  prev = warning('query');
  warning('off');
loaded = load(m{1},var);
  warning(prev);
if isempty(fieldnames(loaded))
  f = [];
  success = 0;
else
  f = loaded.(var);
  success = 1;
end
