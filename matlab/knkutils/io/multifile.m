function f = multifile(files,fun,dim)

% function f = multifile(files,fun,dim)
%
% <files> is a pattern that matches files (see matchfiles.m)
% <fun> is a function to apply to each file
% <dim> is the dimension along which to concatenate results
%
% apply <fun> to each file matched by <files> and concatenate
% the results together along <dim>.
%
% we issue a warning if no files are named by <files>.
%
% example:
% multifile('*',@(x) {x},1)

% transform
files = matchfiles(files);

% check sanity
if length(files)==0
  warning('no file matches');
  f = [];
  return;
end

% do it
f = [];
for p=1:length(files)
  f = cat(dim,f,feval(fun,files{p}));
end
