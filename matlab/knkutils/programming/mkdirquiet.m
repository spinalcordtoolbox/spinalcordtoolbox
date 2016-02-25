function mkdirquiet(x)

% function mkdirquiet(x)
%
% <x> refers to a directory location
%
% make the directory, suppressing warnings
% (e.g. that the directory already exists).
% assert that the result was successful.
%
% example:
% mkdirquiet('temp');

  prev = warning('query'); warning('off');
success = mkdir(x);
  warning(prev);
assert(success,sprintf('mkdir of %s failed',x));
