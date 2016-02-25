function f = checkmemoryworkspace(x)

% function f = checkmemoryworkspace(x)
%
% <x> (optional) is a variable name
% 
% figure out the number of megabytes used in the workspace of
% the caller of this function and return that number.
% if <x> is supplied, we return the size of only that variable.
% if <x> is [] or not supplied, we return the size of the
% entire workspace.
%
% example:
% a = zeros(10000,1000);
% b = zeros(10000,1000);
% checkmemoryworkspace
% checkmemoryworkspace('a')

% input
if ~exist('x','var') || isempty(x)
  x = [];
end

% do it
a = evalin('caller','whos');
if isempty(x)
  f = sum(cat(2,a.bytes))/1024/1024;
else
  ii = find(ismember(cat(2,{a.name}),x));
  f = a(ii).bytes/1024/1024;
end
