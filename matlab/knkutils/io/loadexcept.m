function f = loadexcept(file,vars,mode)

% function f = loadexcept(file,vars,mode)
%
% <file> is a string referring to a .mat file
% <vars> is a variable name or a cell vector of variable names to NOT load
% <mode> (optional) is
%   0 means load <file> into the base workspace (as well as into struct <f>)
%   1 means load <file> into struct <f>
%   Default: 0.
%
% load <file>, excluding variables named by <vars>.
%
% example:
% x = 1; y = 2;
% save('temp.mat','x','y');
% clear x y;
% loadexcept('temp.mat','x')
% ~exist('x','var')
% exist('y','var')

% input
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end
if ~iscell(vars)
  vars = {vars};
end

% figure out variable names
varlist = whos('-file',file);
varlist = cat(2,{varlist.name});

% exclude the ones we don't want
ok = cellfun(@(x) ~ismember(x,vars),varlist);
varlist = varlist(ok);

% load in the data
f = load(file,varlist{:});

% deal
switch mode
case 0

  % assign to caller's workspace
  for p=1:length(varlist)
    assignin('base',varlist{p},f.(varlist{p}));
  end

case 1

end
