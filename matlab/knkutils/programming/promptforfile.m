function file = promptforfile

% function file = promptforfile
%
% prompt the user for a .mat file name.
% check that the file name consists of all word characters
% and that [file name + '.mat'] does not already exist as a file.
% if any of these conditions fail, we automatically re-prompt the user.
% this function is useful for getting a .mat file that we can save into.
%
% return a .mat file name (e.g. 'run1.mat').
%
% example:
% file = promptforfile

while 1
  in = input('please enter name of a .mat file (e.g. ''run1'' means run1.mat): ','s');
  if isempty(regexp(in,'^\w+$'))
    fprintf('invalid name. try again.\n');
  else
    file = [in '.mat'];
    if exist(file,'file')
      fprintf('%s already exists. try again.\n',file);
    else
      break;
    end
  end
end
