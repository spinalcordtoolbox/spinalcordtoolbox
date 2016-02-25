function [f,file] = stripfile(x,flag,sep)

% function [f,file] = stripfile(x,flag,sep)
%
% <x> is a string referring to a file (it is okay if the file
%   does not actually exist).  if <x> ends in /, we automatically
%   act as if that / does not exist.
% <flag> (optional) is whether to swap the output arguments.  default: 0.
% <sep> (optional) is the actual "/" character to use.  default is 
%   the output of filesep.m.
%
% if <flag> is 0,
%   return <f> as the string but with the file name removed.
%   return <file> with the file name.
% if <flag> is 1, these arguments are swapped.
%
% history:
% 2014/07/14 - make more general by defaulting <sep> to filesep.m.
%
% example:
% isequal(stripfile('blah/temp.png',[],'/'),'blah/')
% isequal(stripfile('temp.png',[],'/'),'')
% isequal(stripfile('ok/blah/',1,'/'),'blah')

% input
if ~exist('flag','var') || isempty(flag)
  flag = 0;
end
if ~exist('sep','var') || isempty(sep)
  sep = filesep;
end

% find any /
locs = strfind(x,sep);

% if none, return ''
if isempty(locs)
  f = '';
  file = x;
% otherwise, return entire string up to the last /
else
  if locs(end)==length(x)  % ignore trailing /
    x = x(1:end-1);
    locs = locs(1:end-1);
  end
  if isempty(locs)
    f = '';
    file = x;
  else
    f = x(1:locs(end));
    file = x(locs(end)+1:end);
  end
end

% swap the output?
if flag
  [f,file] = swap(f,file);
end
