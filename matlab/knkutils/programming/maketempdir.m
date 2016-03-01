function f = maketempdir

% function f = maketempdir
%
% make a new empty temporary directory and return the path to it.
%
% example:
% maketempdir

isbad = 1;
while isbad
  f = [tempdir randomword(3) filesep];
  isbad = exist(f,'dir');
end
mkdirquiet(f);
