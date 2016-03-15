function assigntobaseworkspace

% function assigntobaseworkspace
%
% assign all variables that exist in the caller of this function
% to the base workspace.
%
% fun = @(x) assigntobaseworkspace;
% feval(fun,3);
% isequal(x,3)

a = evalin('caller','whos');
for p=1:length(a)
  assignin('base',a(p).name,evalin('caller',a(p).name));
end
