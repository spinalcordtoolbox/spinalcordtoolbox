function y = mergestructs(x,y)

% function y = mergestructs(x,y)
%
% <x>,<y> are structs
%
% make it such that <y> inherits all fields that are defined in <x>.
%
% example:
% x = struct('a',2);
% y = struct('a',1,'b',3);
% mergestructs(x,y)

fields = fieldnames(x);
for p=1:length(fields)
  y.(fields{p}) = x.(fields{p});
end
