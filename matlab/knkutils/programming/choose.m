function f = choose(flag,yes,no)

% function f = choose(flag,yes,no)
%
% <flag> is a truth value (0 or 1)
% <yes> is something
% <no> is something
%
% if <flag>, return <yes>.  otherwise, return <no>.
%
% example:
% isequal(cellfun(@(x) choose(isempty(x),2,x),{[] 1}),[2 1])

if flag
  f = yes;
else
  f = no;
end
