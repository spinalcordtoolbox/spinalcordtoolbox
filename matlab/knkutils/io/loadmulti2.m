function varargout = loadmulti2(varargin)

% function [a,b,c,...] = loadmulti2(file,a,b,c,...)
%
% <file> is a pattern that matches exactly one .mat file (see matchfiles.m)
% <a>,<b>,<c>,... are zero or more variable names
%
% load and return variables from <file>.
% we issue a warning if no file is named by <m>.
%
% example:
% a = 1; b = 2;
% save('atest1.mat','a','b');
% [b2,a2] = loadmulti2('atest1.mat','b','a');
% a==a2 & b==b2

% transform
m = matchfiles(varargin{1});

% check sanity
assert(length(m)==1,'<file> does not match exactly one file');

% get values from the file
varargout = cell(1,length(varargin)-1);
dataload = load(m{1},varargin{2:end});
for p=2:length(varargin)
  varargout{p-1} = dataload.(varargin{p});
end
