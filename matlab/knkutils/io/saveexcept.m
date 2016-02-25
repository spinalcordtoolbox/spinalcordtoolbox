function saveexcept(file,vars)

% function saveexcept(file,vars)
%
% <file> is a string referring to a .mat file
% <vars> is a variable name or a cell vector of variable names to NOT save
%
% save all variables that exist in the caller to <file>, 
% except variables named by <vars>.
%
% example:
% x = 1; y = 2; z = 3;
% saveexcept('temp.mat','z');
% a = load('temp.mat')

% input
if ~iscell(vars)
  vars = {vars};
end

% figure out variable names
varlist = evalin('caller','whos');
varlist = cat(2,{varlist.name});

% exclude the ones we don't want
ok = cellfun(@(x) ~ismember(x,vars),varlist);
varlist = varlist(ok);

% save the data
temp = cell2str(varlist);
cmd = sprintf('save ''%s'' %s;',file,temp(3:end-2));
evalin('caller',cmd);
